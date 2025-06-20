// Package slackbot implements the Slack integration for the MCP client
// It provides event handling, message processing, and integration with LLM services
package slackbot

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"
	"github.com/slack-go/slack/socketmode"
	"github.com/tmc/langchaingo/llms"

	"github.com/tuannvm/slack-mcp-client/internal/common"
	customErrors "github.com/tuannvm/slack-mcp-client/internal/common/errors"
	"github.com/tuannvm/slack-mcp-client/internal/common/logging"
	"github.com/tuannvm/slack-mcp-client/internal/config"
	"github.com/tuannvm/slack-mcp-client/internal/handlers"
	"github.com/tuannvm/slack-mcp-client/internal/llm"
	"github.com/tuannvm/slack-mcp-client/internal/mcp"
	"github.com/tuannvm/slack-mcp-client/internal/slack/formatter"
)

const thinkingMessage = "> Thinking..."

// Client represents the Slack client application.
type Client struct {
	logger          *logging.Logger // Structured logger
	api             *slack.Client
	Socket          *socketmode.Client
	botUserID       string
	botMentionRgx   *regexp.Regexp
	mcpClients      map[string]*mcp.Client
	llmMCPBridge    *handlers.LLMMCPBridge
	llmRegistry     *llm.ProviderRegistry // LLM provider registry
	cfg             *config.Config        // Holds the application configuration
	messageHistory  map[string][]llms.MessageContent
	historyLimit    int
	discoveredTools map[string]common.ToolInfo
	llmsTools       []llms.Tool
}

// NewClient creates a new Slack client instance.
func NewClient(botToken, appToken string, stdLogger *logging.Logger, mcpClients map[string]*mcp.Client,
	discoveredTools map[string]common.ToolInfo, llmsTools []llms.Tool, cfg *config.Config) (*Client, error) {
	if botToken == "" {
		return nil, fmt.Errorf("SLACK_BOT_TOKEN must be set")
	}
	if appToken == "" {
		return nil, fmt.Errorf("SLACK_APP_TOKEN must be set")
	}
	if !strings.HasPrefix(appToken, "xapp-") {
		return nil, fmt.Errorf("SLACK_APP_TOKEN must have the prefix \"xapp-\"")
	}
	// MCP clients are now optional - if none are provided, we'll just use LLM capabilities
	if mcpClients == nil {
		mcpClients = make(map[string]*mcp.Client)
		stdLogger.Printf("No MCP clients provided, running in LLM-only mode")
	}
	if cfg == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}

	// Determine log level from environment variable
	logLevel := logging.LevelInfo // Default to INFO
	if envLevel := os.Getenv("LOG_LEVEL"); envLevel != "" {
		logLevel = logging.ParseLevel(envLevel)
		stdLogger.InfoKV("Setting Slack client log level from environment", "level", envLevel)
	}

	// Create a structured logger for the Slack client
	slackLogger := logging.New("slack-client", logLevel)

	// Initialize the API client
	api := slack.New(
		botToken,
		slack.OptionAppLevelToken(appToken),
		// Still using standard logger for Slack API as it expects a standard logger
		slack.OptionLog(slackLogger.StdLogger()),
	)

	// Authenticate with Slack
	authTest, err := api.AuthTestContext(context.Background())
	if err != nil {
		return nil, customErrors.WrapSlackError(err, "authentication_failed", "Failed to authenticate with Slack")
	}

	// Create the socket mode client
	client := socketmode.New(
		api,
		// Still using standard logger for socket mode as it expects a standard logger
		socketmode.OptionLog(slackLogger.StdLogger()),
		socketmode.OptionDebug(false),
	)

	mentionRegex := regexp.MustCompile(fmt.Sprintf("<@%s>", authTest.UserID))

	// --- MCP/Bridge setup ---
	slackLogger.Printf("Available MCP servers (%d):", len(mcpClients))
	for name := range mcpClients {
		slackLogger.Printf("- %s", name)
	}

	slackLogger.Printf("Available tools (%d):", len(discoveredTools))
	for toolName, toolInfo := range discoveredTools {
		slackLogger.Printf("- %s (Desc: %s, Schema: %v, Server: %s)",
			toolName, toolInfo.Tool.Function.Description, toolInfo.Tool.Function.Parameters, toolInfo.ServerName)
	}

	// Create a map of raw clients to pass to the bridge
	rawClientMap := make(map[string]interface{})
	for name, client := range mcpClients {
		rawClientMap[name] = client
		slackLogger.DebugKV("Adding MCP client to raw map for bridge", "name", name)
	}

	// Determine log level from environment variable
	logLevel = logging.LevelInfo // Default to INFO
	if envLevel := os.Getenv("LOG_LEVEL"); envLevel != "" {
		logLevel = logging.ParseLevel(envLevel)
	}

	// Pass the raw map to the bridge with the configured log level
	llmMCPBridge := handlers.NewLLMMCPBridgeFromClientsWithLogLevel(rawClientMap, slackLogger.StdLogger(), discoveredTools, logLevel)
	slackLogger.InfoKV("LLM-MCP bridge initialized", "clients", len(mcpClients), "tools", len(discoveredTools))

	// --- Initialize the LLM provider registry using the config ---
	// Use the internal structured logger for the registry with the same log level as the bridge
	registryLogger := logging.New("llm-registry", logLevel)
	registry, err := llm.NewProviderRegistry(cfg, registryLogger)
	if err != nil {
		// Log the error using the structured logger
		slackLogger.ErrorKV("Failed to initialize LLM provider registry", "error", err)
		return nil, customErrors.WrapLLMError(err, "llm_registry_init_failed", "Failed to initialize LLM provider registry")
	}
	slackLogger.Info("LLM provider registry initialized successfully")
	// Set the primary provider
	primaryProvider := cfg.LLMProvider
	if primaryProvider == "" {
		slackLogger.Warn("No LLM provider specified in config, using default")
		primaryProvider = "openai" // Default to OpenAI if not specified
	}
	slackLogger.InfoKV("Primary LLM provider set", "provider", primaryProvider)

	// --- Create and return Client instance ---
	return &Client{
		logger:          slackLogger,
		api:             api,
		Socket:          client,
		botUserID:       authTest.UserID,
		botMentionRgx:   mentionRegex,
		mcpClients:      mcpClients,
		llmMCPBridge:    llmMCPBridge,
		llmRegistry:     registry,
		cfg:             cfg,
		messageHistory:  make(map[string][]llms.MessageContent),
		historyLimit:    50, // Store up to 50 messages per channel
		discoveredTools: discoveredTools,
		llmsTools:       llmsTools,
	}, nil
}

// Run starts the Socket Mode event loop and event handling.
func (c *Client) Run() error {
	go c.handleEvents()
	c.logger.Info("Starting Slack Socket Mode listener...")
	return c.Socket.Run()
}

// handleEvents listens for incoming events and dispatches them.
func (c *Client) handleEvents() {
	for evt := range c.Socket.Events {
		switch evt.Type {
		case socketmode.EventTypeConnecting:
			c.logger.Info("Connecting to Slack...")
		case socketmode.EventTypeConnectionError:
			c.logger.Warn("Connection failed. Retrying...")
		case socketmode.EventTypeConnected:
			c.logger.Info("Connected to Slack!")
		case socketmode.EventTypeEventsAPI:
			eventsAPIEvent, ok := evt.Data.(slackevents.EventsAPIEvent)
			if !ok {
				c.logger.WarnKV("Ignored unexpected EventsAPI event type", "type", fmt.Sprintf("%T", evt.Data))
				continue
			}
			c.Socket.Ack(*evt.Request)
			c.logger.InfoKV("Received EventsAPI event", "type", eventsAPIEvent.Type)
			c.handleEventMessage(eventsAPIEvent)
		default:
			c.logger.DebugKV("Ignored event type", "type", evt.Type)
		}
	}
	c.logger.Info("Slack event channel closed.")
}

// handleEventMessage processes specific EventsAPI messages.
func (c *Client) handleEventMessage(event slackevents.EventsAPIEvent) {
	switch event.Type {
	case slackevents.CallbackEvent:
		innerEvent := event.InnerEvent
		switch ev := innerEvent.Data.(type) {
		case *slackevents.AppMentionEvent:
			timestamp := ev.ThreadTimeStamp
			if timestamp == "" {
				timestamp = ev.TimeStamp // Use the message timestamp if no thread
			}
			c.logger.InfoKV("Received app mention in channel", "channel", ev.Channel, "user", ev.User, "text", ev.Text, "timestamp", timestamp)
			messageText := c.botMentionRgx.ReplaceAllString(ev.Text, "")
			// Add to message history
			c.addToHistory(timestamp, llms.ChatMessageTypeHuman, llms.TextPart(messageText))
			// Use handleUserPrompt for app mentions too, for consistency
			go c.handleUserPrompt(strings.TrimSpace(messageText), ev.Channel, timestamp)

		case *slackevents.MessageEvent:
			isDirectMessage := strings.HasPrefix(ev.Channel, "D")
			isValidUser := ev.User != "" && ev.User != c.botUserID
			isNotEdited := ev.SubType != "message_changed"
			isBot := ev.BotID != "" || ev.SubType == "bot_message"

			if isDirectMessage && isValidUser && isNotEdited && !isBot {
				timestamp := ev.ThreadTimeStamp
				if timestamp == "" {
					timestamp = ev.TimeStamp // Use the message timestamp if no thread
				}
				c.logger.InfoKV("Received direct message in channel", "channel", ev.Channel, "user", ev.User, "text", ev.Text, "timestamp", timestamp)
				// Add to message history
				c.addToHistory(timestamp, llms.ChatMessageTypeHuman, llms.TextPart(ev.Text))
				go c.handleUserPrompt(ev.Text, ev.Channel, timestamp) // Use goroutine to avoid blocking event loop
			}

		default:
			c.logger.DebugKV("Unsupported inner event type", "type", fmt.Sprintf("%T", innerEvent.Data))
		}
	default:
		c.logger.DebugKV("Unsupported outer event type", "type", event.Type)
	}
}

// addToHistory adds a message to the channel history
func (c *Client) addToHistory(threadTS string, role llms.ChatMessageType, parts ...llms.ContentPart) {
	history, exists := c.messageHistory[threadTS]
	if !exists {
		history = []llms.MessageContent{}
	}

	// Add the new message
	message := llms.MessageContent{
		Role:  role,
		Parts: parts,
	}
	history = append(history, message)

	// Limit history size
	if len(history) > c.historyLimit {
		history = history[len(history)-c.historyLimit:]
	}

	c.messageHistory[threadTS] = history
}

// getContextFromHistory builds a context string from message history
//
//nolint:unused // Reserved for future use
func (c *Client) getContextFromHistory(threadTS string) []llms.MessageContent {
	c.logger.DebugKV("Built conversation context", "thread", threadTS)
	history := c.messageHistory[threadTS]
	return history
}

// handleUserPrompt sends the user's text to the configured LLM provider.
func (c *Client) handleUserPrompt(userPrompt, channelID, threadTS string) {
	// Determine the provider to use from config
	c.logger.DebugKV("User prompt", "text", userPrompt)

	c.addToHistory(threadTS, llms.ChatMessageTypeHuman, llms.TextPart(userPrompt)) // Add user message to history

	// Process the LLM response through the MCP pipeline
	c.processLLMResponseAndReply(channelID, threadTS)
}

// callLLM generates a text completion using the specified provider from the registry.
func (c *Client) callLLM(providerName string, messages []llms.MessageContent) (*llms.ContentChoice, error) {
	// Create a context with appropriate timeout
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// Generate the system prompt with tool information
	//toolPrompt := c.generateToolPrompt()

	// Build options based on the config (provider might override or use these)
	// Note: TargetProvider is removed as it's handled by config/factory
	options := llm.ProviderOptions{
		// Model: Let the provider use its configured default or handle overrides if needed.
		// Model: c.cfg.OpenAIModelName, // Example: If you still want a global default hint
		Temperature: 0.7,         // Consider making configurable
		MaxTokens:   2048,        // Consider making configurable
		Tools:       c.llmsTools, // Use the tools registered in the client
	}

	// --- Use the specified provider via the registry ---
	c.logger.InfoKV("Attempting to use LLM provider for chat completion", "provider", providerName)

	// Call the registry's method which includes availability check
	resp, err := c.llmRegistry.GenerateChatCompletion(ctx, providerName, messages, options)
	if err != nil {
		// Error already logged by registry method potentially, but log here too for context
		c.logger.ErrorKV("GenerateChatCompletion failed", "provider", providerName, "error", err)
		return nil, customErrors.WrapSlackError(err, "llm_request_failed", fmt.Sprintf("LLM request failed for provider '%s'", providerName))
	}

	c.logger.InfoKV("Successfully received chat completion", "provider", providerName)
	return resp, nil
}

func (c *Client) answer(channelID, threadTS, respTimestamp string, response string) {
	if response == "" {
		response = "(LLM returned an empty response)"
	}

	c.postMessage(channelID, threadTS, respTimestamp, response)
}

// processLLMResponseAndReply processes the LLM response, handles tool results with re-prompting, and sends the final reply.
// Incorporates logic previously in LLMClient.ProcessToolResponse.
func (c *Client) processLLMResponseAndReply(channelID, threadTS string) {
	providerName := c.cfg.LLMProvider

	for {
		// Show a temporary "typing" indicator
		_, messageTimestamp, err := c.api.PostMessage(channelID, slack.MsgOptionText(thinkingMessage, false), slack.MsgOptionTS(threadTS))
		if err != nil {
			c.logger.ErrorKV("Error posting typing indicator", "error", err)
		}

		// Get context from history
		contextHistory := c.getContextFromHistory(threadTS)
		// Call LLM using the integrated logic
		llmResponse, err := c.callLLM(providerName, contextHistory)
		if err != nil {
			c.logger.ErrorKV("Error from LLM provider", "provider", providerName, "error", err)
			c.answer(channelID, threadTS, messageTimestamp, fmt.Sprintf("Sorry, I encountered an error with the LLM provider ('%s'): %v", providerName, err))
			return
		}

		c.logger.InfoKV("Received response from LLM", "provider", providerName, "length", len(llmResponse.Content))

		// Log the raw LLM response for debugging
		c.logger.DebugKV("Raw LLM response", "response", truncateForLog(llmResponse.Content, 500))

		// Add the LLM response to history

		// End of the response handling logic
		if len(llmResponse.ToolCalls) == 0 || c.llmMCPBridge == nil {
			if c.llmMCPBridge == nil {
				c.logger.Warn("LLMMCPBridge is nil, skipping tool processing")
			}
			c.addToHistory(threadTS, llms.ChatMessageTypeAI, llms.TextPart(llmResponse.Content))

			c.answer(channelID, threadTS, messageTimestamp, llmResponse.Content)
			return
		}

		parts := []llms.ContentPart{
			llms.TextPart(llmResponse.Content),
		}
		for _, toolCall := range llmResponse.ToolCalls {
			parts = append(parts, toolCall)
		}

		c.addToHistory(threadTS, llms.ChatMessageTypeAI, parts...)

		// Create a context with timeout for tool processing
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		for _, toolCall := range llmResponse.ToolCalls {
			messageParts := []string{}
			if llmResponse.Content != "" {
				messageParts = append(messageParts, llmResponse.Content)
			}
			toolCallMessage := fmt.Sprintf("calling tool `%s` with arguments: `%s`", toolCall.FunctionCall.Name, toolCall.FunctionCall.Arguments)
			messageParts = append(messageParts, toolCallMessage)

			message := "> " + strings.Join(messageParts, " - ")
			fmt.Println("Tool call message:", message)
			c.answer(channelID, threadTS, messageTimestamp, message)

			// --- Process Tool Response (Logic from LLMClient.ProcessToolResponse) ---
			// Process the response through the bridge
			args := make(map[string]interface{})
			err = json.Unmarshal([]byte(toolCall.FunctionCall.Arguments), &args)
			if err != nil {
				c.logger.ErrorKV("Failed to unmarshal tool call arguments", "error", err, "arguments", toolCall.FunctionCall.Arguments)
				return
			}

			processedResponse, err := c.llmMCPBridge.ProcessLLMResponse(ctx, toolCall.FunctionCall.Name, args)
			if err != nil {
				c.logger.ErrorKV("Tool processing error", "error", err)
				finalResponse := fmt.Sprintf("Sorry, I encountered an error while trying to use a tool: %v", err)
				c.answer(channelID, "", messageTimestamp, finalResponse)
				return
			}

			c.logger.DebugKV("Tool result", "result", truncateForLog(processedResponse, 500))

			// Construct a new prompt incorporating the original prompt and the tool result
			//rePrompt := fmt.Sprintf("The user asked: '%s'\n\nI used a tool and received the following result:\n```\n%s\n```\nPlease formulate a concise and helpful natural language response to the user based *only* on the user's original question and the tool result provided.", userPrompt, finalResponse)

			// Add history
			toolResponsePart := llms.ToolCallResponse{
				ToolCallID: toolCall.ID,
				Name:       toolCall.FunctionCall.Name,
				Content:    processedResponse,
			}
			c.addToHistory(threadTS, llms.ChatMessageTypeTool, toolResponsePart)
		}
	}
}

// truncateForLog truncates a string for log output
func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// postMessage sends a message back to Slack, replying in a thread if threadTS is provided.
func (c *Client) postMessage(channelID, threadTS, respTimestamp, text string) {
	if text == "" {
		c.logger.WarnKV("Attempted to send empty message, skipping", "channel", channelID)
		return
	}

	// Detect message type and format accordingly
	messageType := formatter.DetectMessageType(text)
	c.logger.DebugKV("Detected message type", "type", messageType, "length", len(text))
	messageType = formatter.PlainText

	var msgOptions []slack.MsgOption
	var formattedText = text

	options := formatter.DefaultOptions()

	switch messageType {
	case formatter.JSONBlock:
		// Message is already in Block Kit JSON format
		options.Format = formatter.BlockFormat
		formattedText = text
	case formatter.StructuredData:
		// Convert structured data to Block Kit format
		formattedText = formatter.FormatStructuredData(text)
		options.Format = formatter.BlockFormat
	case formatter.MarkdownText:
		formattedText = formatter.FormatMarkdown(text)
	}

	if threadTS != "" {
		options.ThreadTS = threadTS
	}
	msgOptions = formatter.FormatMessage(formattedText, options)

	// Send the message
	var err error
	_, _, _, err = c.api.UpdateMessage(channelID, respTimestamp, msgOptions...)

	if err != nil {
		c.logger.ErrorKV("Error posting message to channel", "channel", channelID, "error", err, "messageType", messageType)

		// If we get an error with Block Kit format, try falling back to plain text
		if messageType == formatter.JSONBlock || messageType == formatter.StructuredData {
			c.logger.InfoKV("Falling back to plain text format due to Block Kit error", "channel", channelID)

			// Apply markdown formatting to the original text and send as plain text
			formattedText := formatter.FormatMarkdown(text)
			fallbackOptions := []slack.MsgOption{
				slack.MsgOptionText(formattedText, false),
			}
			if threadTS != "" {
				fallbackOptions = append(fallbackOptions, slack.MsgOptionTS(threadTS))
			}

			// Try sending with plain text format
			var fallbackErr error
			_, _, _, fallbackErr = c.api.UpdateMessage(channelID, respTimestamp, fallbackOptions...)

			if fallbackErr != nil {
				c.logger.ErrorKV("Error posting fallback message to channel", "channel", channelID, "error", fallbackErr)
			}
		}
	}
}
