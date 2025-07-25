"""Tests for the server functions in the SageMaker AI MCP Server."""

import pytest
from sagemaker_ai_mcp_server.server import (
    create_mlflow_tracking_server_sagemaker,
    create_presigned_url_for_domain_sagemaker,
    create_presigned_url_for_mlflow_tracking_server_sagemaker,
    delete_domain_sagemaker,
    delete_endpoint_config_sagemaker,
    delete_endpoint_sagemaker,
    delete_mlflow_tracking_server_sagemaker,
    delete_pipeline_sagemaker,
    describe_domain_sagemaker,
    describe_endpoint_config_sagemaker,
    describe_endpoint_sagemaker,
    describe_mlflow_tracking_server_sagemaker,
    describe_pipeline_definition_for_execution_sagemaker,
    describe_pipeline_execution_sagemaker,
    describe_pipeline_sagemaker,
    describe_processing_job_sagemaker,
    describe_training_job_sagemaker,
    describe_transform_job_sagemaker,
    list_domains_sagemaker,
    list_endpoint_configs_sagemaker,
    list_endpoints_sagemaker,
    list_mlflow_tracking_servers_sagemaker,
    list_pipeline_execution_steps_sagemaker,
    list_pipeline_executions_sagemaker,
    list_pipeline_parameters_for_execution_sagemaker,
    list_pipelines_sagemaker,
    list_processing_jobs_sagemaker,
    list_spaces_sagemaker,
    list_training_jobs_sagemaker,
    list_transform_jobs_sagemaker,
    list_user_profiles_sagemaker,
    start_mlflow_tracking_server_sagemaker,
    stop_mlflow_tracking_server_sagemaker,
    stop_processing_job_sagemaker,
    stop_training_job_sagemaker,
    stop_transform_job_sagemaker,
)
from unittest.mock import patch


@pytest.mark.asyncio
async def test_list_endpoints_sagemaker():
    """Test the list_endpoints_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_endpoints') as mock_list_endpoints:
        mock_list_endpoints.return_value = [{'EndpointName': 'test-endpoint'}]

        result = await list_endpoints_sagemaker()

        mock_list_endpoints.assert_called_once()
        assert result == {'endpoints': [{'EndpointName': 'test-endpoint'}]}


@pytest.mark.asyncio
async def test_list_endpoint_configs_sagemaker():
    """Test the list_endpoint_configs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_endpoint_configs') as mock_list_configs:
        mock_list_configs.return_value = [{'EndpointConfigName': 'test-config'}]

        result = await list_endpoint_configs_sagemaker()

        mock_list_configs.assert_called_once()
        assert result == {'endpoint_configs': [{'EndpointConfigName': 'test-config'}]}


@pytest.mark.asyncio
async def test_delete_endpoint_sagemaker():
    """Test the delete_endpoint_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_endpoint') as mock_delete_endpoint:
        endpoint_name = 'test-endpoint'
        result = await delete_endpoint_sagemaker(endpoint_name)

        mock_delete_endpoint.assert_called_once_with(endpoint_name)
        expected_msg = f"Endpoint '{endpoint_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_delete_endpoint_config_sagemaker():
    """Test the delete_endpoint_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_endpoint_config') as mock_delete_config:
        config_name = 'test-endpoint-config'

        result = await delete_endpoint_config_sagemaker(config_name)

        mock_delete_config.assert_called_once_with(config_name)
        expected_msg = f"Endpoint Config '{config_name}' deleted successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_describe_endpoint_sagemaker():
    """Test the describe_endpoint_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_endpoint') as mock_describe_endpoint:
        endpoint_name = 'test-endpoint'
        expected_result = {
            'EndpointName': endpoint_name,
            'EndpointStatus': 'InService',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_endpoint.return_value = expected_result

        result = await describe_endpoint_sagemaker(endpoint_name)

        mock_describe_endpoint.assert_called_once_with(endpoint_name)
        assert result == {'endpoint_details': expected_result}


@pytest.mark.asyncio
async def test_describe_endpoint_config_sagemaker():
    """Test the describe_endpoint_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_endpoint_config') as mock_describe_config:
        config_name = 'test-endpoint-config'
        expected_result = {
            'EndpointConfigName': config_name,
            'CreationTime': '2023-01-01T00:00:00',
            'ProductionVariants': [{'VariantName': 'test-variant'}],
        }
        mock_describe_config.return_value = expected_result

        result = await describe_endpoint_config_sagemaker(config_name)

        mock_describe_config.assert_called_once_with(config_name)
        assert result == {'endpoint_config_details': expected_result}


@pytest.mark.asyncio
async def test_describe_training_job_sagemaker():
    """Test the describe_training_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_training_job') as mock_describe_job:
        job_name = 'test-training-job'
        expected_result = {
            'TrainingJobName': job_name,
            'TrainingJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_job.return_value = expected_result

        result = await describe_training_job_sagemaker(job_name)

        mock_describe_job.assert_called_once_with(job_name)
        assert result == {'training_job_details': expected_result}


@pytest.mark.asyncio
async def test_list_training_jobs_sagemaker():
    """Test the list_training_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_training_jobs') as mock_list_jobs:
        mock_list_jobs.return_value = [
            {'TrainingJobName': 'test-job-1'},
            {'TrainingJobName': 'test-job-2'},
        ]

        result = await list_training_jobs_sagemaker()

        mock_list_jobs.assert_called_once()
        assert result == {
            'training_jobs': [{'TrainingJobName': 'test-job-1'}, {'TrainingJobName': 'test-job-2'}]
        }


@pytest.mark.asyncio
async def test_stop_training_job_sagemaker():
    """Test the stop_training_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_training_job') as mock_stop_job:
        job_name = 'test-training-job'
        await stop_training_job_sagemaker(job_name)

        mock_stop_job.assert_called_once_with(job_name)
        expected_msg = f"Training job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_processing_jobs_sagemaker():
    """Test the list_processing_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_processing_jobs') as mock_list_processing:
        mock_list_processing.return_value = [
            {'ProcessingJobName': 'test-processing-job-1'},
            {'ProcessingJobName': 'test-processing-job-2'},
        ]

        result = await list_processing_jobs_sagemaker()

        mock_list_processing.assert_called_once()
        assert result == {
            'processing_jobs': [
                {'ProcessingJobName': 'test-processing-job-1'},
                {'ProcessingJobName': 'test-processing-job-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_processing_job_sagemaker():
    """Test the describe_processing_job_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_processing_job'
    ) as mock_describe_processing:
        job_name = 'test-processing-job'
        expected_result = {
            'ProcessingJobName': job_name,
            'ProcessingJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_processing.return_value = expected_result

        result = await describe_processing_job_sagemaker(job_name)

        mock_describe_processing.assert_called_once_with(job_name)
        assert result == {'processing_job_details': expected_result}


@pytest.mark.asyncio
async def test_stop_processing_job_sagemaker():
    """Test the stop_processing_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_processing_job') as mock_stop_processing:
        job_name = 'test-processing-job'
        await stop_processing_job_sagemaker(job_name)

        mock_stop_processing.assert_called_once_with(job_name)
        expected_msg = f"Processing job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_transform_jobs_sagemaker():
    """Test the list_transform_jobs_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_transform_jobs') as mock_list_transform:
        mock_list_transform.return_value = [
            {'TransformJobName': 'test-transform-job-1'},
            {'TransformJobName': 'test-transform-job-2'},
        ]

        result = await list_transform_jobs_sagemaker()

        mock_list_transform.assert_called_once()
        assert result == {
            'transform_jobs': [
                {'TransformJobName': 'test-transform-job-1'},
                {'TransformJobName': 'test-transform-job-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_transform_job_sagemaker():
    """Test the describe_transform_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_transform_job') as mock_describe_transform:
        job_name = 'test-transform-job'
        expected_result = {
            'TransformJobName': job_name,
            'TransformJobStatus': 'Completed',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_transform.return_value = expected_result

        result = await describe_transform_job_sagemaker(job_name)

        mock_describe_transform.assert_called_once_with(job_name)
        assert result == {'transform_job_details': expected_result}


@pytest.mark.asyncio
async def test_stop_transform_job_sagemaker():
    """Test the stop_transform_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_transform_job') as mock_stop_transform:
        job_name = 'test-transform-job'
        await stop_transform_job_sagemaker(job_name)

        mock_stop_transform.assert_called_once_with(job_name)
        expected_msg = f"Transform job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_pipeline_executions_sagemaker():
    """Test the list_pipeline_executions_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_pipeline_executions') as mock_list_executions:
        mock_list_executions.return_value = [
            {
                'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution-1'
            },
            {
                'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution-2'
            },
        ]

        result = await list_pipeline_executions_sagemaker('test-pipeline')

        mock_list_executions.assert_called_once_with('test-pipeline')
        assert result == {
            'pipeline_executions': [
                {
                    'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution-1'
                },
                {
                    'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution-2'
                },
            ]
        }


@pytest.mark.asyncio
async def test_list_pipeline_execution_steps_sagemaker():
    """Test the list_pipeline_execution_steps_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_pipeline_execution_steps') as mock_list_steps:
        mock_list_steps.return_value = [
            {'StepName': 'test-step-1'},
            {'StepName': 'test-step-2'},
        ]

        result = await list_pipeline_execution_steps_sagemaker('test-pipeline')

        mock_list_steps.assert_called_once_with('test-pipeline')
        assert result == {
            'pipeline_execution_steps': [
                {'StepName': 'test-step-1'},
                {'StepName': 'test-step-2'},
            ]
        }


@pytest.mark.asyncio
async def test_list_pipeline_parameters_for_execution_sagemaker():
    """Test the list_pipeline_parameters_for_execution_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_pipeline_parameters_for_execution'
    ) as mock_list_params:
        mock_list_params.return_value = [
            {'Name': 'param1', 'Value': 'value1'},
            {'Name': 'param2', 'Value': 'value2'},
        ]

        result = await list_pipeline_parameters_for_execution_sagemaker('test-pipeline')

        mock_list_params.assert_called_once_with('test-pipeline')
        assert result == {
            'pipeline_parameters': [
                {'Name': 'param1', 'Value': 'value1'},
                {'Name': 'param2', 'Value': 'value2'},
            ]
        }


@pytest.mark.asyncio
async def test_list_pipelines_sagemaker():
    """Test the list_pipelines_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_pipelines') as mock_list_pipelines:
        mock_list_pipelines.return_value = [
            {'PipelineName': 'test-pipeline-1'},
            {'PipelineName': 'test-pipeline-2'},
        ]

        result = await list_pipelines_sagemaker()

        mock_list_pipelines.assert_called_once()
        assert result == {
            'pipelines': [
                {'PipelineName': 'test-pipeline-1'},
                {'PipelineName': 'test-pipeline-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_pipeline_sagemaker():
    """Test the describe_pipeline_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_pipeline') as mock_describe_pipeline:
        pipeline_name = 'test-pipeline'
        expected_result = {
            'PipelineName': pipeline_name,
            'PipelineArn': f'arn:aws:sagemaker:us-west-2:123456789012:pipeline/{pipeline_name}',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_pipeline.return_value = expected_result

        result = await describe_pipeline_sagemaker(pipeline_name)

        mock_describe_pipeline.assert_called_once_with(pipeline_name)
        assert result == {'pipeline_details': expected_result}


@pytest.mark.asyncio
async def test_delete_pipeline_sagemaker():
    """Test the delete_pipeline_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_pipeline') as mock_delete_pipeline:
        pipeline_name = 'test-pipeline'
        await delete_pipeline_sagemaker(pipeline_name)

        mock_delete_pipeline.assert_called_once_with(pipeline_name)
        expected_msg = f"Pipeline '{pipeline_name}' deleted successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_describe_pipeline_definition_for_execution_sagemaker():
    """Test the describe_pipeline_definition_for_execution_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_pipeline_definition_for_execution'
    ) as mock_describe_definition:
        execution_arn = 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
        expected_result = {
            'PipelineDefinition': 'test-definition',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_definition.return_value = expected_result

        result = await describe_pipeline_definition_for_execution_sagemaker(execution_arn)

        mock_describe_definition.assert_called_once_with(execution_arn)
        assert result == {'pipeline_definition': expected_result}


@pytest.mark.asyncio
async def test_describe_pipeline_execution_sagemaker():
    """Test the describe_pipeline_execution_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_pipeline_execution'
    ) as mock_describe_execution:
        execution_arn = 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
        expected_result = {
            'PipelineExecutionStatus': 'InProgress',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_execution.return_value = expected_result

        result = await describe_pipeline_execution_sagemaker(execution_arn)

        mock_describe_execution.assert_called_once_with(execution_arn)
        assert result == {'pipeline_execution_details': expected_result}


@pytest.mark.asyncio
async def test_create_mlflow_tracking_server_sagemaker():
    """Test the create_mlflow_tracking_server_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.create_mlflow_tracking_server'
    ) as mock_create_server:
        server_name = 'test-mlflow-server'
        artifact_uri = 's3://test-bucket/artifacts'
        server_size = 'Medium'
        msg = f"MLflow Tracking Server '{server_name}' created successfully"

        result = await create_mlflow_tracking_server_sagemaker(
            tracking_server_name=server_name,
            artifact_store_uri=artifact_uri,
            tracking_server_size=server_size,
        )

        mock_create_server.assert_called_once_with(server_name, artifact_uri, server_size)
        assert result == {'message': msg}


@pytest.mark.asyncio
async def test_delete_mlflow_tracking_server_sagemaker():
    """Test the delete_mlflow_tracking_server_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.delete_mlflow_tracking_server'
    ) as mock_delete_server:
        server_name = 'test-mlflow-server'
        msg = f"MLflow Tracking Server '{server_name}' deleted successfully"

        result = await delete_mlflow_tracking_server_sagemaker(server_name)

        mock_delete_server.assert_called_once_with(server_name)
        assert result == {'message': msg}


@pytest.mark.asyncio
async def test_list_mlflow_tracking_servers_sagemaker():
    """Test the list_mlflow_tracking_servers_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_mlflow_tracking_servers') as mock_list_servers:
        mock_list_servers.return_value = [
            {'TrackingServerName': 'test-mlflow-server-1'},
            {'TrackingServerName': 'test-mlflow-server-2'},
        ]

        result = await list_mlflow_tracking_servers_sagemaker()

        mock_list_servers.assert_called_once()
        assert result == {
            'tracking_servers': [
                {'TrackingServerName': 'test-mlflow-server-1'},
                {'TrackingServerName': 'test-mlflow-server-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_mlflow_tracking_server_sagemaker():
    """Test the describe_mlflow_tracking_server_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_mlflow_tracking_server'
    ) as mock_describe_server:
        server_name = 'test-mlflow-server'
        arn_base = 'arn:aws:sagemaker:us-west-2:123456789012'
        server_arn = f'{arn_base}:mlflow-tracking-server/{server_name}'
        expected_result = {
            'TrackingServerName': server_name,
            'TrackingServerArn': server_arn,
            'TrackingServerStatus': 'InService',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_server.return_value = expected_result

        result = await describe_mlflow_tracking_server_sagemaker(server_name)

        mock_describe_server.assert_called_once_with(server_name)
        assert result == {'tracking_server_details': expected_result}


@pytest.mark.asyncio
async def test_create_presigned_url_for_mlflow_tracking_server_sagemaker():
    """Test the create_presigned_url function for MLflow tracking server."""
    with patch(
        'sagemaker_ai_mcp_server.server.create_presigned_mlflow_tracking_server_url'
    ) as mock_create_url:
        server_name = 'test-mlflow-server'
        expiration = 3600
        url = 'https://test-presigned-url.aws.com'
        mock_create_url.return_value = url

        func = create_presigned_url_for_mlflow_tracking_server_sagemaker
        result = await func(tracking_server_name=server_name, expiration_seconds=expiration)

        mock_create_url.assert_called_once_with(server_name, expiration)
        assert result == {'presigned_url': url}


@pytest.mark.asyncio
async def test_start_mlflow_tracking_server_sagemaker():
    """Test the start_mlflow_tracking_server_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.start_mlflow_tracking_server') as mock_start_server:
        server_name = 'test-mlflow-server'
        msg = f"MLflow Tracking Server '{server_name}' started successfully"

        result = await start_mlflow_tracking_server_sagemaker(server_name)

        mock_start_server.assert_called_once_with(server_name)
        assert result == {'message': msg}


@pytest.mark.asyncio
async def test_stop_mlflow_tracking_server_sagemaker():
    """Test the stop_mlflow_tracking_server_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_mlflow_tracking_server') as mock_stop_server:
        server_name = 'test-mlflow-server'
        msg = f"MLflow Tracking Server '{server_name}' stopped successfully"

        result = await stop_mlflow_tracking_server_sagemaker(server_name)

        mock_stop_server.assert_called_once_with(server_name)
        assert result == {'message': msg}


@pytest.mark.asyncio
async def test_delete_domain_sagemaker():
    """Test the delete_domain_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_domain') as mock_delete_domain:
        domain_id = 'test-domain'
        await delete_domain_sagemaker(domain_id)

        mock_delete_domain.assert_called_once_with(domain_id)
        expected_msg = f"Domain '{domain_id}' deleted successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_domains_sagemaker():
    """Test the list_domains_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_domains') as mock_list_domains:
        mock_list_domains.return_value = [{'DomainId': 'test-domain'}]

        result = await list_domains_sagemaker()

        mock_list_domains.assert_called_once()
        assert result == {'domains': [{'DomainId': 'test-domain'}]}


@pytest.mark.asyncio
async def test_describe_domain_sagemaker():
    """Test the describe_domain_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_domain') as mock_describe_domain:
        domain_id = 'test-domain'
        expected_result = {
            'DomainId': domain_id,
            'DomainName': 'Test Domain',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_domain.return_value = expected_result

        result = await describe_domain_sagemaker(domain_id)

        mock_describe_domain.assert_called_once_with(domain_id)
        assert result == {'domain_details': expected_result}


@pytest.mark.asyncio
async def test_create_presigned_url_for_domain_sagemaker():
    """Test the create_presigned_url_for_domain_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.create_presigned_domain_url') as mock_create_url:
        domain_id = 'test-domain'
        expiration = 3600
        user_profile_name = 'test-user-profile'
        url = 'https://example.com/presigned-domain-url'
        mock_create_url.return_value = url

        result = await create_presigned_url_for_domain_sagemaker(
            domain_id=domain_id, user_profile_name=user_profile_name, expiration_seconds=expiration
        )

        mock_create_url.assert_called_once_with(domain_id, user_profile_name, expiration)
        assert result == {'presigned_url': url}


@pytest.mark.asyncio
async def test_list_spaces_sagemaker():
    """Test the list_spaces_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_spaces') as mock_list_spaces:
        mock_list_spaces.return_value = [{'SpaceName': 'test-space'}]

        result = await list_spaces_sagemaker()

        mock_list_spaces.assert_called_once()
        assert result == {'spaces': [{'SpaceName': 'test-space'}]}


@pytest.mark.asyncio
async def test_list_user_profiles_sagemaker():
    """Test the list_user_profiles_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_user_profiles') as mock_list_user_profiles:
        mock_list_user_profiles.return_value = [{'UserProfileName': 'test-user-profile'}]

        result = await list_user_profiles_sagemaker()

        mock_list_user_profiles.assert_called_once()
        assert result == {'user_profiles': [{'UserProfileName': 'test-user-profile'}]}
