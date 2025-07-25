"""The helper functions for the SageMaker AI MCP Server."""

import boto3
import os
from loguru import logger
from typing import Any, Dict, List, Literal


# Helper Functions for the region, session and client creation


def get_region() -> str:
    """Get the AWS region from the environment variable or default to 'us-east-1'.

    Returns:
        str: The AWS region.
    """
    return os.getenv('AWS_REGION', 'us-east-1')


def get_aws_session(region_name=None) -> boto3.Session:
    """Create an AWS session using AWS Profile or default credentials.

    Args:
        region_name (str): The AWS region to use. Defaults to None, which uses the
                           region from the environment variable or defaults to 'us-east-1'.

    Returns:
        boto3.Session: An AWS session object.
    """
    profile_name = os.environ.get('AWS_PROFILE')
    region = region_name or get_region()
    try:
        if profile_name:
            logger.debug(f'Using AWS profile: {profile_name}')
            return boto3.Session(profile_name=profile_name, region_name=region)
        else:
            logger.debug('Using default AWS credential chain')
            return boto3.Session(region_name=region)
    except Exception as e:
        logger.error(f'Error creating AWS session: {e}')
        raise RuntimeError(f'Failed to create AWS session: {e}')


def get_sagemaker_client(region_name=None):
    """Get a SageMaker client.

    Args:
        region_name (str): The AWS region to use. Defaults to None, which uses the
                           region from the environment variable or defaults to 'us-east-1'.

    Returns:
        boto3.client: A SageMaker client object.
    """
    session = get_aws_session(region_name)
    return session.client('sagemaker')


def get_sagemaker_execution_role_arn() -> str:
    """Get the SageMaker execution role ARN from the environment variable.

    Returns:
        str: The SageMaker execution role ARN.
    """
    role_arn = os.environ.get('SAGEMAKER_EXECUTION_ROLE_ARN')
    if not role_arn:
        raise ValueError('SAGEMAKER_EXECUTION_ROLE_ARN environment variable is not set.')
    return role_arn


# Helper Functions for SageMaker AI Endpoint Ops


async def list_endpoints() -> List[Dict[str, Any]]:
    """List all SageMaker Endpoints that are available.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Endpoints.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Endpoints...')
    response = client.list_endpoints()
    return response.get('Endpoints', [])


async def list_endpoint_configs() -> List[Dict[str, Any]]:
    """List all SageMaker Endpoint Configurations.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Endpoint Configurations.
    """
    client = get_sagemaker_client()
    response = client.list_endpoint_configs()
    logger.info('Listing SageMaker Endpoint Configurations...')
    return response.get('EndpointConfigs', [])


async def delete_endpoint(endpoint_name: str) -> None:
    """Delete a SageMaker Endpoint.

    Args:
        endpoint_name (str): The name of the SageMaker Endpoint to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Endpoint: {endpoint_name}')
    client.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f'Endpoint {endpoint_name} deleted successfully.')


async def delete_endpoint_config(endpoint_config_name: str) -> None:
    """Delete a SageMaker Endpoint Configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker Endpoint Configuration to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Endpoint Config: {endpoint_config_name}')
    client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    logger.info(f'Endpoint Config {endpoint_config_name} deleted successfully.')


async def describe_endpoint(endpoint_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Endpoint.

    Args:
        endpoint_name (str): The name of the SageMaker Endpoint to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Endpoint.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Endpoint: {endpoint_name}')
    response = client.describe_endpoint(EndpointName=endpoint_name)
    return response


async def describe_endpoint_config(endpoint_config_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Endpoint Configuration.

    Args:
        endpoint_config_name (str): The name of the SageMaker Endpoint Configuration to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Endpoint Configuration.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Endpoint Config: {endpoint_config_name}')
    response = client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    return response


# Helper Functions for SageMaker AI Training, Processing, Transform Jobs


async def list_training_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Training Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Training Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Training Jobs...')
    response = client.list_training_jobs()
    return response.get('TrainingJobSummaries', [])


async def describe_training_job(training_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Training Job."""
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Training Job: {training_job_name}')
    response = client.describe_training_job(TrainingJobName=training_job_name)
    return response


async def stop_training_job(training_job_name: str) -> None:
    """Stop a SageMaker Training Job.

    Args:
        training_job_name (str): The name of the SageMaker Training Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Training Job: {training_job_name}')
    client.stop_training_job(TrainingJobName=training_job_name)
    logger.info(f'Training Job {training_job_name} stopped successfully.')


async def list_processing_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Processing Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Processing Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Processing Jobs...')
    response = client.list_processing_jobs()
    return response.get('ProcessingJobSummaries', [])


async def describe_processing_job(processing_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Processing Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Processing Job: {processing_job_name}')
    response = client.describe_processing_job(ProcessingJobName=processing_job_name)
    return response


async def stop_processing_job(processing_job_name: str) -> None:
    """Stop a SageMaker Processing Job.

    Args:
        processing_job_name (str): The name of the SageMaker Processing Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Processing Job: {processing_job_name}')
    client.stop_processing_job(ProcessingJobName=processing_job_name)
    logger.info(f'Processing Job {processing_job_name} stopped successfully.')


async def list_transform_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Transform Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Transform Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Transform Jobs...')
    response = client.list_transform_jobs()
    return response.get('TransformJobSummaries', [])


async def describe_transform_job(transform_job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Transform Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Transform Job: {transform_job_name}')
    response = client.describe_transform_job(TransformJobName=transform_job_name)
    return response


async def stop_transform_job(transform_job_name: str) -> None:
    """Stop a SageMaker Transform Job.

    Args:
        transform_job_name (str): The name of the SageMaker Transform Job to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Transform Job: {transform_job_name}')
    client.stop_transform_job(TransformJobName=transform_job_name)


# Helper Functions for SageMaker AI Pipeline Ops


async def list_pipelines() -> List[Dict[str, Any]]:
    """List all SageMaker Pipelines.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Pipelines.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Pipelines...')
    response = client.list_pipelines()
    return response.get('PipelineSummaries', [])


async def describe_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Pipeline.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Pipeline: {pipeline_name}')
    response = client.describe_pipeline(PipelineName=pipeline_name)
    return response


async def delete_pipeline(pipeline_name: str) -> None:
    """Delete a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Pipeline: {pipeline_name}')
    client.delete_pipeline(PipelineName=pipeline_name)
    logger.info(f'Pipeline {pipeline_name} deleted successfully.')


async def list_pipeline_parameters_for_execution(
    pipeline_execution_arn: str,
) -> List[Dict[str, Any]]:
    """List parameters for a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        List[Dict[str, Any]]: A list of parameters for the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing parameters for Pipeline Execution: {pipeline_execution_arn}')
    response = client.list_pipeline_parameters_for_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )
    return response.get('PipelineParameters', [])


async def list_pipeline_executions(pipeline_name: str) -> List[Dict[str, Any]]:
    """List all executions of a specific SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline.

    Returns:
        List[Dict[str, Any]]: A list of Pipeline Executions for the specified Pipeline.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing executions for Pipeline: {pipeline_name}')
    response = client.list_pipeline_executions(PipelineName=pipeline_name)
    return response.get('PipelineExecutionSummaries', [])


async def list_pipeline_execution_steps(
    pipeline_execution_arn: str,
) -> List[Dict[str, Any]]:
    """List steps for a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        List[Dict[str, Any]]: A list of steps for the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing steps for Pipeline Execution: {pipeline_execution_arn}')
    response = client.list_pipeline_execution_steps(PipelineExecutionArn=pipeline_execution_arn)
    return response.get('PipelineExecutionSteps', [])


async def describe_pipeline_execution(
    pipeline_execution_arn: str,
) -> Dict[str, Any]:
    """Describe a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        Dict[str, Any]: The details of the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing Pipeline Execution: {pipeline_execution_arn}')
    response = client.describe_pipeline_execution(PipelineExecutionArn=pipeline_execution_arn)
    return response


async def describe_pipeline_definition_for_execution(
    pipeline_execution_arn: str,
) -> Dict[str, Any]:
    """Describe the definition of a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution.

    Returns:
        Dict[str, Any]: The definition of the specified Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing Pipeline Definition for Execution: {pipeline_execution_arn}')
    response = client.describe_pipeline_definition_for_execution(
        PipelineExecutionArn=pipeline_execution_arn
    )
    return response


async def start_pipeline_execution(
    pipeline_name: str,
    pipeline_parameters: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Start a new execution of a SageMaker Pipeline.

    Args:
        pipeline_name (str): The name of the SageMaker Pipeline.
        pipeline_parameters (List[Dict[str, Any]], optional): Parameters for the Pipeline Execution. Defaults to None.

    Returns:
        Dict[str, Any]: The details of the started Pipeline Execution.
    """
    client = get_sagemaker_client()
    logger.info(f'Starting Pipeline Execution for: {pipeline_name}')
    response = client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=pipeline_parameters or [],
    )
    return response


async def stop_pipeline_execution(pipeline_execution_arn: str) -> None:
    """Stop a specific SageMaker Pipeline Execution.

    Args:
        pipeline_execution_arn (str): The ARN of the SageMaker Pipeline Execution to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping Pipeline Execution: {pipeline_execution_arn}')
    client.stop_pipeline_execution(PipelineExecutionArn=pipeline_execution_arn)
    logger.info(f'Pipeline Execution {pipeline_execution_arn} stopped successfully.')


# Helper Functions for SageMaker AI MLflow Managed Tracking Server Ops


async def create_mlflow_tracking_server(
    tracking_server_name: str,
    artifact_store_uri: str,
    tracking_server_size: Literal['Small', 'Medium', 'Large'],
) -> str:
    """Create an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server.
        artifact_store_uri (str): The S3 URI for the artifact store.
        tracking_server_size (Literal['Small', 'Medium', 'Large']): The size of the Tracking Server.

    Returns:
        str: The ARN of the created MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    role_arn = get_sagemaker_execution_role_arn()
    logger.info(f'Creating MLflow Tracking Server: {tracking_server_name}')
    response = client.create_mlflow_tracking_server(
        TrackingServerName=tracking_server_name,
        ArtifactStoreUri=artifact_store_uri,
        TrackingServerSize=tracking_server_size,
        RoleArn=role_arn,
    )
    logger.info(f'MLflow Tracking Server {tracking_server_name} created successfully.')
    return response.get('TrackingServerArn', '')


async def delete_mlflow_tracking_server(tracking_server_name: str) -> None:
    """Delete an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting MLflow Tracking Server: {tracking_server_name}')
    client.delete_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    logger.info(f'MLflow Tracking Server {tracking_server_name} deleted successfully.')


async def list_mlflow_tracking_servers() -> List[Dict[str, Any]]:
    """List all MLflow Tracking Servers in SageMaker.

    Returns:
        List[Dict[str, Any]]: A list of MLflow Tracking Servers.
    """
    client = get_sagemaker_client()
    logger.info('Listing MLflow Tracking Servers...')
    response = client.list_mlflow_tracking_servers()
    return response.get('TrackingServerSummaries', [])


async def describe_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Describe a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to describe.

    Returns:
        Dict[str, Any]: The details of the specified MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing MLflow Tracking Server: {tracking_server_name}')
    response = client.describe_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def start_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Start a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to start.

    Returns:
        Dict[str, Any]: The details of the started MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Starting MLflow Tracking Server: {tracking_server_name}')
    response = client.start_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def stop_mlflow_tracking_server(
    tracking_server_name: str,
) -> Dict[str, Any]:
    """Stop a specific MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server to stop.

    Returns:
        Dict[str, Any]: The details of the stopped MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping MLflow Tracking Server: {tracking_server_name}')
    response = client.stop_mlflow_tracking_server(TrackingServerName=tracking_server_name)
    return response


async def create_presigned_mlflow_tracking_server_url(
    tracking_server_name: str,
    expiration_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing an MLflow Tracking Server in SageMaker.

    Args:
        tracking_server_name (str): The name of the MLflow Tracking Server.
        expiration_seconds (int): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the MLflow Tracking Server.
    """
    client = get_sagemaker_client()
    logger.info(f'Creating presigned URL for MLflow Tracking Server: {tracking_server_name}')
    response = client.create_presigned_mlflow_tracking_server_url(
        TrackingServerName=tracking_server_name,
        ExpirationSeconds=expiration_seconds,
    )
    return response.get('PresignedUrl', '')


# Helper Functions for SageMaker AI Domain Ops


async def delete_domain(domain_id: str) -> None:
    """Delete a SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Domain: {domain_id}')
    client.delete_domain(DomainId=domain_id)
    logger.info(f'Domain {domain_id} deleted successfully.')


async def list_domains() -> List[Dict[str, Any]]:
    """List all SageMaker Domains.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Domains.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Domains...')
    response = client.list_domains()
    return response.get('Domains', [])


async def describe_domain(domain_id: str) -> Dict[str, Any]:
    """Describe a specific SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain to describe.

    Returns:
        Dict[str, Any]: The details of the specified SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Domain: {domain_id}')
    response = client.describe_domain(DomainId=domain_id)
    return response


async def create_presigned_domain_url(
    domain_id: str,
    user_profile_name: str,
    expiration_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing a SageMaker Domain.

    Args:
        domain_id (str): The ID of the SageMaker Domain.
        user_profile_name (str): The name of the user profile.
        expiration_seconds (int): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info(f'Creating presigned URL for SageMaker Domain: {domain_id}')
    response = client.create_presigned_domain_url(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        ExpirationSeconds=expiration_seconds,
    )
    return response.get('AuthorizedUrl', '')


# Helper Functions for SageMaker AI User Profile Ops


async def list_user_profiles() -> List[Dict[str, Any]]:
    """List all user profiles in a SageMaker Domain.

    Returns:
        List[Dict[str, Any]]: A list of user profiles in the SageMaker Domain.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker User Profiles...')
    response = client.list_user_profiles()
    return response.get('UserProfiles', [])


async def list_spaces() -> List[Dict[str, Any]]:
    """List all SageMaker Spaces.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Spaces.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Spaces...')
    response = client.list_spaces()
    return response.get('Spaces', [])


# Helper Functions for SageMaker AI Model Ops


async def describe_model(model_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Model.

    Args:
        model_name (str): The name of the SageMaker Model to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Model.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Model: {model_name}')
    response = client.describe_model(ModelName=model_name)
    return response


async def delete_model(model_name: str) -> None:
    """Delete a SageMaker Model.

    Args:
        model_name (str): The name of the SageMaker Model to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Model: {model_name}')
    client.delete_model(ModelName=model_name)
    logger.info(f'Model {model_name} deleted successfully.')


async def list_models() -> List[Dict[str, Any]]:
    """List all SageMaker Models.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Models.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Models...')
    response = client.list_models()
    return response.get('Models', [])


# Helper Functions for SageMaker AI Model Cards


async def describe_model_card(model_card_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker Model Card.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Model Card: {model_card_name}')
    response = client.describe_model_card(ModelCardName=model_card_name)
    return response


async def list_model_cards() -> List[Dict[str, Any]]:
    """List all SageMaker Model Cards.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Model Cards.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Model Cards...')
    response = client.list_model_cards()
    return response.get('ModelCardSummaries', [])


async def delete_model_card(model_card_name: str) -> None:
    """Delete a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker Model Card: {model_card_name}')
    client.delete_model_card(ModelCardName=model_card_name)
    logger.info(f'Model Card {model_card_name} deleted successfully.')


async def list_model_card_export_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Model Card Export Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Model Card Export Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Model Card Export Jobs...')
    response = client.list_model_card_export_jobs()
    return response.get('ModelCardExportJobSummaries', [])


async def list_model_card_versions(model_card_name: str) -> List[Dict[str, Any]]:
    """List all versions of a SageMaker Model Card.

    Args:
        model_card_name (str): The name of the SageMaker Model Card.

    Returns:
        List[Dict[str, Any]]: A list of versions of the specified Model Card.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing versions for Model Card: {model_card_name}')
    response = client.list_model_card_versions(ModelCardName=model_card_name)
    return response.get('ModelCardVersionSummaryList', [])


# Helper Functions for SageMaker AI Inference Recommender Jobs


async def list_inference_recommendations_jobs() -> List[Dict[str, Any]]:
    """List all SageMaker Inference Recommender Jobs.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Inference Recommender Jobs.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Inference Recommender Jobs...')
    response = client.list_inference_recommendations_jobs()
    return response.get('InferenceRecommendationsJobs', [])


async def list_inference_recommendations_job_steps(job_name: str) -> List[Dict[str, Any]]:
    """List steps for a specific SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job.

    Returns:
        List[Dict[str, Any]]: A list of steps for the specified Inference
        Recommender Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Listing steps for Inference Recommender Job: {job_name}')
    response = client.list_inference_recommendations_job_steps(JobName=job_name)
    return response.get('Steps', [])


async def stop_inference_recommendations_job(job_name: str) -> None:
    """Stop a SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job
            to stop.
    """
    client = get_sagemaker_client()
    logger.info(f'Stopping SageMaker Inference Recommender Job: {job_name}')
    client.stop_inference_recommendations_job(JobName=job_name)
    logger.info(f'Inference Recommender Job {job_name} stopped successfully.')


async def describe_inference_recommendations_job(job_name: str) -> Dict[str, Any]:
    """Describe a SageMaker Inference Recommender Job.

    Args:
        job_name (str): The name of the SageMaker Inference Recommender Job
            to describe.

    Returns:
        Dict[str, Any]: The details of the specified Inference Recommender Job.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker Inference Recommender Job: {job_name}')
    response = client.describe_inference_recommendations_job(JobName=job_name)
    return response


# Helper Functions for SageMaker AI Apps Operations


async def create_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
    resource_spec: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which to create the app.
        user_profile_name (str): The name of the user profile in which to create the app.
        app_type (str): The type of app to create, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app.
        resource_spec (Dict[str, Any], optional): The resource specification for the app. Defaults to None.

    Returns:
        Dict[str, Any]: The details of the created SageMaker App.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Creating SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )

    create_params = {
        'DomainId': domain_id,
        'UserProfileName': user_profile_name,
        'AppType': app_type,
        'AppName': app_name,
    }

    if resource_spec:
        create_params['ResourceSpec'] = resource_spec

    response = client.create_app(**create_params)
    logger.info(f'App {app_name} creation initiated successfully.')
    return response.get('AppArn', {})


async def delete_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
) -> None:
    """Delete a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which the app resides.
        user_profile_name (str): The name of the user profile that owns the app.
        app_type (str): The type of app to delete, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app to delete.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Deleting SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )
    client.delete_app(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )
    logger.info(f'App {app_name} deletion initiated successfully.')


async def delete_app_image_config(app_image_config_name: str) -> None:
    """Delete a SageMaker App Image Config.

    Args:
        app_image_config_name (str): The name of the SageMaker App Image Config to delete.
    """
    client = get_sagemaker_client()
    logger.info(f'Deleting SageMaker App Image Config: {app_image_config_name}')
    client.delete_app_image_config(AppImageConfigName=app_image_config_name)
    logger.info(f'App Image Config {app_image_config_name} deleted successfully.')


async def describe_app(
    domain_id: str,
    user_profile_name: str,
    app_type: Literal[
        'JupyterServer',
        'KernelGateway',
        'RStudioServerPro',
        'RSessionGateway',
        'Canvas',
        'JupyterLab',
        'CodeEditor',
        'TensorBoard',
        'DetailedProfiler',
    ],
    app_name: str,
) -> Dict[str, Any]:
    """Describe a SageMaker App.

    Args:
        domain_id (str): The ID of the domain in which the app resides.
        user_profile_name (str): The name of the user profile that owns the app.
        app_type (str): The type of app, e.g., 'JupyterServer', 'KernelGateway'.
        app_name (str): The name of the app.

    Returns:
        Dict[str, Any]: The details of the SageMaker App.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Describing SageMaker App: {app_name} of type {app_type} for user {user_profile_name} in domain {domain_id}'
    )
    response = client.describe_app(
        DomainId=domain_id,
        UserProfileName=user_profile_name,
        AppType=app_type,
        AppName=app_name,
    )
    return response


async def describe_app_image_config(app_image_config_name: str) -> Dict[str, Any]:
    """Describe a SageMaker App Image Config.

    Args:
        app_image_config_name (str): The name of the SageMaker App Image Config to describe.

    Returns:
        Dict[str, Any]: The details of the SageMaker App Image Config.
    """
    client = get_sagemaker_client()
    logger.info(f'Describing SageMaker App Image Config: {app_image_config_name}')
    response = client.describe_app_image_config(AppImageConfigName=app_image_config_name)
    return response


async def list_apps() -> List[Dict[str, Any]]:
    """List all SageMaker Apps.

    Returns:
        List[Dict[str, Any]]: A list of SageMaker Apps.
    """
    client = get_sagemaker_client()
    logger.info('Listing SageMaker Apps...')

    response = client.list_apps()
    return response.get('Apps', [])


async def create_presigned_notebook_instance_url(
    notebook_instance_name: str,
    session_expiration_duration_in_seconds: int = 3600,
) -> str:
    """Create a presigned URL for accessing a SageMaker Notebook Instance.

    Args:
        notebook_instance_name (str): The name of the SageMaker Notebook Instance.
        session_expiration_duration_in_seconds (int, optional): The expiration time for the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The presigned URL for the SageMaker Notebook Instance.
    """
    client = get_sagemaker_client()
    logger.info(
        f'Creating presigned URL for SageMaker Notebook Instance: {notebook_instance_name}'
    )
    response = client.create_presigned_notebook_instance_url(
        NotebookInstanceName=notebook_instance_name,
        SessionExpirationDurationInSeconds=session_expiration_duration_in_seconds,
    )
    return response.get('AuthorizedUrl', '')
