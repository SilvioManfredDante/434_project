package main

import (
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/sns"
)

func main() {
	// Get the AWS region and SNS topic ARN from environment variables
	region := "us-east-1"  // Change to your preferred region
	topicArn := os.Getenv("SNS_TOPIC_ARN")
	if topicArn == "" {
		log.Fatal("SNS_TOPIC_ARN environment variable not set")
	}

	// Initialize a session in the region
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String(region),
	})
	if err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	// Create an SNS client
	svc := sns.New(sess)

	// The message to send
	message := "Hello from the Go microservice on AWS!"

	// Publish the message to SNS
	_, err = svc.Publish(&sns.PublishInput{
		Message:  aws.String(message),
		TopicArn: aws.String(topicArn),
	})
	if err != nil {
		log.Fatalf("failed to publish message: %v", err)
	}

	fmt.Println("Message sent to SNS topic successfully!")
}
