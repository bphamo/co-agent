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
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	// Parse API key from .env and load into the anthropic instance
	apiKey := os.Getenv("ANTHROPIC_API_KEY")

	if apiKey == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY is not set")
		return
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
	err = agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}
