# SageMaker AI MCP Server

A Model Context Protocol (MCP) server for Amazon SageMaker AI that enables AI assistants to access, work and manage SageMaker AI resources.

## Features

- Managing and working with SageMaker AI endpoint resources
- Managing and working with SageMaker AI training, processing and transform jobs
- Managing and working with SageMaker AI pipelines
- CRUD operations for SageMaker AI Managed MLflow Tracking Server
- CRUD operations for SageMaker AI Domain
- Managing and working with Models and Model Cards
- CRUD operatioons for SageMaker AI Recommender Jobs
- CRUD operations for SageMaker AI Apps

## Prerequisites

1. Install `uv` from [Astral](https://docs.astral.sh/uv/getting-started/installation/) or the [GitHub README](https://github.com/astral-sh/uv#installation)
2. Install Python using `uv python install 3.10`
3. Set up AWS credentials with access to Amazon SageMaker AI resources.
   - You need an AWS account with Amazon SageMaker AI enabled
   - Configure AWS credentials with `aws configure` or environment variables
   - Ensure your IAM role/user has permissions to use Amazon SageMaker AI
4. Create a SageMaker Execution Role with the necessary permissions for SageMaker AI operations

## Installation

WIP

## Environment Variables

- `AWS_PROFILE`: AWS CLI profile to use for credentials
- `AWS_REGION`: AWS region to use (default: us-east-1)
- `SAGEMAKER_EXECUTION_ROLE_ARN`: ARN of the SageMaker execution role

## AWS Authentication

The server uses the AWS profile specified in the `AWS_PROFILE` environment variable. If not provided, it defaults to the default credential provider chain.

```json
"env": {
  "AWS_PROFILE": "your-aws-profile",
  "AWS_REGION": "us-east-1",
  "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
}
```

Make sure the AWS profile has permissions to access Amazon SageMaker AI services. The MCP server creates a boto3 session using the specified profile to authenticate with AWS services.

## Tools

WIP

## Example Usage

WIP

## Security Considerations

- Use AWS IAM roles with appropriate permissions
- Store credentials securely
- Use temporary credentials when possible

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
