package main

import (
	"agent/agents"
	"agent/tools"
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

func main() {
	// A .env file is optional: the API key can also come from the real
	// environment (containers, CI, an exported shell var). godotenv does not
	// overwrite variables that are already set, so the environment wins.
	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		log.Fatalf("Error loading .env file: %s", err)
	}

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("Error: ANTHROPIC_API_KEY is not set")
	}

	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []tools.ToolDefinition{tools.ReadFileDefinition}

	agent := agents.NewAgent(&client, getUserMessage, tools, anthropic.ModelClaude3_7SonnetLatest)
	if err := agent.Run(context.TODO()); err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}
