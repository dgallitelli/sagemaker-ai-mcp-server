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

### List of Tools for SageMaker AI Endpoints and Endpoint Configurations
- list_endpoints_sagemaker (List all SageMaker AI Endpoints)
- list_endpoint_configs_sagemaker (List all SageMaker AI Endpoint Configurations)
- describe_endpoint_sagemaker (Describe a SageMaker AI Endpoint)
- describe_endpoint_config_sagemaker (Describe a SageMaker AI Endpoint Configuration)
- delete_endpoint_sagemaker (Delete a SageMaker AI Endpoint)
- delete_endpoint_config_sagemaker (Delete a SageMaker AI Endpoint Configuration)

### List of Tools for SageMaker AI Jobs
- list_training_jobs_sagemaker (List all SageMaker AI Training Jobs)
- list_processing_jobs_sagemaker (List all SageMaker AI Processing Jobs)
- list_transform_jobs_sagemaker (List all SageMaker AI Transform Jobs)
- list_inference_recommender_jobs_sagemaker (List all SageMaker AI Inference Recommender Jobs)
- list_inference_recommender_job_steps_sagemaker (List all steps for a SageMaker AI Inference Recommender Job)
- describe_training_job_sagemaker (Describe a SageMaker AI Training Job)
- describe_processing_job_sagemaker (Describe a SageMaker AI Processing Job)
- describe_transform_job_sagemaker (Describe a SageMaker AI Transform Job)
- describe_inference_recommender_job_sagemaker (Describe a SageMaker AI Inference Recommender Job)
- stop_training_job_sagemaker (Stop a SageMaker AI Training Job)
- stop_processing_job_sagemaker (Stop a SageMaker AI Processing Job)
- stop_transform_job_sagemaker (Stop a SageMaker AI Transform Job)
- stop_inference_recommender_job_sagemaker (Stop a SageMaker AI Inference Recommender Job)

### List of Tools for SageMaker AI Pipelines
- list_pipelines_sagemaker (List all SageMaker AI Pipelines)
- list_pipeline_executions_sagemaker (List all Pipeline Executions for a SageMaker AI Pipeline)
- list_pipeline_execution_steps_sagemaker (List all steps for a SageMaker AI Pipeline Execution)
- list_pipeline_parameters_for_execution_sagemaker (List all parameters for a SageMaker AI Pipeline Execution)
- describe_pipeline_sagemaker (Describe a SageMaker AI Pipeline)
- describe_pipeline_execution_sagemaker (Describe a SageMaker AI Pipeline Execution)
- describe_pipeline_definition_for_execution_sagemaker (Describe a SageMaker AI Pipeline Definition for Execution)
- start_pipeline_execution_sagemaker (Start a SageMaker AI Pipeline Execution)
- stop_pipeline_execution_sagemaker (Stop a SageMaker AI Pipeline Execution)
- delete_pipeline_sagemaker (Delete a SageMaker AI Pipeline)

### List of Tools for SageMaker AI User Profiles and Spaces
- list_user_profiles_sagemaker (List all SageMaker AI User Profiles)
- list_spaces_sagemaker (List all SageMaker AI Spaces)

### List of Tools for SageMaker AI MLflow Managed Tracking Servers
- list_mlflow_tracking_servers_sagemaker (List all MLflow Tracking Servers)
- create_mlflow_tracking_server_sagemaker (Create a new MLflow Tracking Server)
- create_presigned_mlflow_tracking_server_url_sagemaker (Create a presigned URL for an MLflow Tracking Server)
- describe_mlflow_tracking_server_sagemaker (Describe an MLflow Tracking Server)
- start_mlflow_tracking_server_sagemaker (Start an MLflow Tracking Server)
- stop_mlflow_tracking_server_sagemaker (Stop an MLflow Tracking Server)
- delete_mlflow_tracking_server_sagemaker (Delete an MLflow Tracking Server)

### List of Tools for SageMaker AI Domains
- list_domains_sagemaker (List all SageMaker AI Domains)
- create_presigned_domain_url_sagemaker (Create a presigned URL for a SageMaker Domain)
- describe_domain_sagemaker (Describe a SageMaker AI Domain)
- delete_domain_sagemaker (Delete a SageMaker AI Domain)

### List of Tools for SageMaker AI Models
- list_models_sagemaker (List all SageMaker AI Models)
- describe_model_sagemaker (Describe a SageMaker AI Model)
- delete_model_sagemaker (Delete a SageMaker AI Model)

### List of Tools for SageMaker AI Model Cards
- list_model_cards_sagemaker (List all SageMaker AI Model Cards)
- list_model_card_export_jobs_sagemaker (List all SageMaker AI Model Card Export Jobs)
- list_model_card_versions_sagemaker (List all versions of a SageMaker AI Model Card)
- describe_model_card_sagemaker (Describe a SageMaker AI Model Card)
- delete_model_card_sagemaker (Delete a SageMaker AI Model Card)

### List of Tools for SageMaker AI Apps
- list_apps_sagemaker (List all SageMaker AI Apps)
- create_app_sagemaker (Create a SageMaker AI App)
- create_presigned_notebook_instance_url_sagemaker (Create a presigned URL for a SageMaker Notebook Instance)
- describe_app_sagemaker (Describe a SageMaker AI App)
- describe_app_image_config_sagemaker (Describe a SageMaker AI App Image Config)
- delete_app_sagemaker (Delete a SageMaker AI App)
- delete_app_image_config_sagemaker (Delete a SageMaker AI App Image Config)

## Security Considerations

- Use AWS IAM roles with appropriate permissions
- Store credentials securely
- Use temporary credentials when possible

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
