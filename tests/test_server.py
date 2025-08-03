"""Tests for the server functions in the SageMaker AI MCP Server."""

import pytest
from sagemaker_ai_mcp_server.server import (
    create_app_sagemaker,
    create_mlflow_tracking_server_sagemaker,
    create_presigned_notebook_instance_url_sagemaker,
    create_presigned_url_for_domain_sagemaker,
    create_presigned_url_for_mlflow_tracking_server_sagemaker,
    delete_app_image_config_sagemaker,
    delete_app_sagemaker,
    delete_domain_sagemaker,
    delete_endpoint_config_sagemaker,
    delete_endpoint_sagemaker,
    delete_mlflow_tracking_server_sagemaker,
    delete_model_card_sagemaker,
    delete_model_sagemaker,
    delete_pipeline_sagemaker,
    describe_app_image_config_sagemaker,
    describe_app_sagemaker,
    describe_domain_sagemaker,
    describe_endpoint_config_sagemaker,
    describe_endpoint_sagemaker,
    describe_inference_recommendations_job_sagemaker,
    describe_mlflow_tracking_server_sagemaker,
    describe_model_card_sagemaker,
    describe_model_sagemaker,
    describe_pipeline_definition_for_execution_sagemaker,
    describe_pipeline_execution_sagemaker,
    describe_pipeline_sagemaker,
    describe_processing_job_sagemaker,
    describe_training_job_sagemaker,
    describe_transform_job_sagemaker,
    list_apps_sagemaker,
    list_domains_sagemaker,
    list_endpoint_configs_sagemaker,
    list_endpoints_sagemaker,
    list_inference_recommendations_job_steps_sagemaker,
    list_inference_recommendations_jobs_sagemaker,
    list_mlflow_tracking_servers_sagemaker,
    list_model_card_export_jobs_sagemaker,
    list_model_card_versions_sagemaker,
    list_model_cards_sagemaker,
    list_models_sagemaker,
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
    start_pipeline_execution_sagemaker,
    stop_inference_recommendations_job_sagemaker,
    stop_mlflow_tracking_server_sagemaker,
    stop_pipeline_execution_sagemaker,
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
async def test_list_inference_recommendations_jobs_sagemaker():
    """Test the list_inference_recommendations_jobs_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_inference_recommendations_jobs'
    ) as mock_list_jobs:
        mock_list_jobs.return_value = [
            {'JobName': 'test-job-1', 'Status': 'Completed'},
            {'JobName': 'test-job-2', 'Status': 'InProgress'},
        ]

        result = await list_inference_recommendations_jobs_sagemaker()

        assert 'inference_recommendations_jobs' in result
        assert len(result['inference_recommendations_jobs']) == 2
        assert result['inference_recommendations_jobs'][0]['JobName'] == 'test-job-1'
        mock_list_jobs.assert_called_once()


@pytest.mark.asyncio
async def test_list_inference_recommendations_job_steps_sagemaker():
    """Test the list_inference_recommendations_job_steps_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_inference_recommendations_job_steps'
    ) as mock_list_steps:
        job_name = 'test-job'
        mock_list_steps.return_value = [
            {'StepName': 'step-1', 'Status': 'Completed'},
            {'StepName': 'step-2', 'Status': 'InProgress'},
        ]

        result = await list_inference_recommendations_job_steps_sagemaker(job_name=job_name)

        assert 'steps' in result
        assert len(result['steps']) == 2
        assert result['steps'][0]['StepName'] == 'step-1'
        mock_list_steps.assert_called_once_with(job_name)


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
async def test_describe_inference_recommendations_job_sagemaker():
    """Test the describe_inference_recommendations_job_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.describe_inference_recommendations_job'
    ) as mock_describe_job:
        job_name = 'test-job'
        mock_describe_job.return_value = {
            'JobName': job_name,
            'Status': 'Completed',
            'JobType': 'Default',
            'CreationTime': '2023-01-01T00:00:00.000Z',
        }

        result = await describe_inference_recommendations_job_sagemaker(job_name=job_name)

        assert 'job_details' in result
        assert result['job_details']['JobName'] == job_name
        assert result['job_details']['Status'] == 'Completed'
        mock_describe_job.assert_called_once_with(job_name)


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
async def test_stop_processing_job_sagemaker():
    """Test the stop_processing_job_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_processing_job') as mock_stop_processing:
        job_name = 'test-processing-job'
        await stop_processing_job_sagemaker(job_name)

        mock_stop_processing.assert_called_once_with(job_name)
        expected_msg = f"Processing job '{job_name}' stopped successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


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
async def test_stop_inference_recommendations_job_sagemaker():
    """Test the stop_inference_recommendations_job_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.stop_inference_recommendations_job'
    ) as mock_stop_job:
        job_name = 'test-job'

        result = await stop_inference_recommendations_job_sagemaker(job_name=job_name)

        assert 'message' in result
        assert f"Inference Recommender Job '{job_name}' stopped successfully" in result['message']
        mock_stop_job.assert_called_once_with(job_name)


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
async def test_start_pipeline_execution_sagemaker():
    """Test the start_pipeline_execution_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.start_pipeline_execution') as mock_start_execution:
        pipeline_name = 'test-pipeline'
        parameters = {'param1': 'value1', 'param2': 'value2'}
        execution_arn = f'arn:aws:sagemaker:us-west-2:123456789012:pipeline/{pipeline_name}/execution/test-execution'
        mock_start_execution.return_value = execution_arn

        result = await start_pipeline_execution_sagemaker(pipeline_name, parameters)

        mock_start_execution.assert_called_once_with(pipeline_name, parameters)
        expected_msg = f"Pipeline '{pipeline_name}' started successfully with ARN: {execution_arn}"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_stop_pipeline_execution_sagemaker():
    """Test the stop_pipeline_execution_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.stop_pipeline_execution') as mock_stop_execution:
        execution_arn = 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'

        result = await stop_pipeline_execution_sagemaker(execution_arn)

        mock_stop_execution.assert_called_once_with(execution_arn)
        expected_msg = f"Pipeline Execution '{execution_arn}' stopped successfully"
        assert result == {'message': expected_msg}


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
async def test_list_user_profiles_sagemaker():
    """Test the list_user_profiles_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_user_profiles') as mock_list_user_profiles:
        mock_list_user_profiles.return_value = [{'UserProfileName': 'test-user-profile'}]

        result = await list_user_profiles_sagemaker()

        mock_list_user_profiles.assert_called_once()
        assert result == {'user_profiles': [{'UserProfileName': 'test-user-profile'}]}


@pytest.mark.asyncio
async def test_list_spaces_sagemaker():
    """Test the list_spaces_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_spaces') as mock_list_spaces:
        mock_list_spaces.return_value = [{'SpaceName': 'test-space'}]

        result = await list_spaces_sagemaker()

        mock_list_spaces.assert_called_once()
        assert result == {'spaces': [{'SpaceName': 'test-space'}]}


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
async def test_create_mlflow_tracking_server_sagemaker():
    """Test the create_mlflow_tracking_server_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.create_mlflow_tracking_server'
    ) as mock_create_server:
        server_name = 'test-mlflow-server'
        artifact_uri = 's3://test-bucket/artifacts'
        server_size = 'Medium'
        mock_create_server.return_value = (
            'arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-mlflow-server'
        )

        result = await create_mlflow_tracking_server_sagemaker(
            tracking_server_name=server_name,
            artifact_store_uri=artifact_uri,
            tracking_server_size=server_size,
        )

        mock_create_server.assert_called_once_with(server_name, artifact_uri, server_size)
        assert result == {'tracking_server_arn': mock_create_server.return_value}


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
async def test_list_domains_sagemaker():
    """Test the list_domains_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_domains') as mock_list_domains:
        mock_list_domains.return_value = [{'DomainId': 'test-domain'}]

        result = await list_domains_sagemaker()

        mock_list_domains.assert_called_once()
        assert result == {'domains': [{'DomainId': 'test-domain'}]}


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
async def test_delete_domain_sagemaker():
    """Test the delete_domain_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_domain') as mock_delete_domain:
        domain_id = 'test-domain'
        await delete_domain_sagemaker(domain_id)

        mock_delete_domain.assert_called_once_with(domain_id)
        expected_msg = f"Domain '{domain_id}' deleted successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_models_sagemaker():
    """Test the list_models_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_models') as mock_list_models:
        mock_list_models.return_value = [
            {'ModelName': 'test-model-1'},
            {'ModelName': 'test-model-2'},
        ]

        result = await list_models_sagemaker()

        mock_list_models.assert_called_once()
        assert result == {
            'models': [
                {'ModelName': 'test-model-1'},
                {'ModelName': 'test-model-2'},
            ]
        }


@pytest.mark.asyncio
async def test_describe_model_sagemaker():
    """Test the describe_model_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_model') as mock_describe_model:
        model_name = 'test-model'
        expected_result = {
            'ModelName': model_name,
            'ModelArn': f'arn:aws:sagemaker:us-west-2:123456789012:model/{model_name}',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_model.return_value = expected_result

        result = await describe_model_sagemaker(model_name)

        mock_describe_model.assert_called_once_with(model_name)
        assert result == {'model_details': expected_result}


@pytest.mark.asyncio
async def test_delete_model_sagemaker():
    """Test the delete_model_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_model') as mock_delete_model:
        model_name = 'test-model'
        await delete_model_sagemaker(model_name)

        mock_delete_model.assert_called_once_with(model_name)
        expected_msg = f"Model '{model_name}' deleted successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_list_model_cards_sagemaker():
    """Test the list_model_cards_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_model_cards') as mock_list_model_cards:
        mock_list_model_cards.return_value = [
            {'ModelCardId': 'test-model-card-1'},
            {'ModelCardId': 'test-model-card-2'},
        ]

        result = await list_model_cards_sagemaker()

        mock_list_model_cards.assert_called_once()
        assert result == {
            'model_cards': [
                {'ModelCardId': 'test-model-card-1'},
                {'ModelCardId': 'test-model-card-2'},
            ]
        }


@pytest.mark.asyncio
async def test_list_model_card_export_jobs_sagemaker():
    """Test the list_model_card_export_jobs_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.list_model_card_export_jobs'
    ) as mock_list_export_jobs:
        mock_list_export_jobs.return_value = [
            {'ModelCardExportJobName': 'test-export-job-1'},
            {'ModelCardExportJobName': 'test-export-job-2'},
        ]

        result = await list_model_card_export_jobs_sagemaker('test-model-card')

        mock_list_export_jobs.assert_called_once_with('test-model-card')
        assert result == {
            'model_card_export_jobs': [
                {'ModelCardExportJobName': 'test-export-job-1'},
                {'ModelCardExportJobName': 'test-export-job-2'},
            ]
        }


@pytest.mark.asyncio
async def test_list_model_card_versions_sagemaker():
    """Test the list_model_card_versions_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_model_card_versions') as mock_list_versions:
        mock_list_versions.return_value = [
            {'ModelCardVersion': 'v1.0'},
            {'ModelCardVersion': 'v1.1'},
        ]

        result = await list_model_card_versions_sagemaker('test-model-card')

        mock_list_versions.assert_called_once_with('test-model-card')
        assert result == {
            'model_card_versions': [
                {'ModelCardVersion': 'v1.0'},
                {'ModelCardVersion': 'v1.1'},
            ]
        }


@pytest.mark.asyncio
async def test_delete_model_card_sagemaker():
    """Test the delete_model_card_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_model_card') as mock_delete_model_card:
        model_card_id = 'test-model-card'
        await delete_model_card_sagemaker(model_card_id)

        mock_delete_model_card.assert_called_once_with(model_card_id)
        expected_msg = f"Model Card '{model_card_id}' deleted successfully"
        assert {'message': expected_msg} == {'message': expected_msg}


@pytest.mark.asyncio
async def test_describe_model_card_sagemaker():
    """Test the describe_model_card_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_model_card') as mock_describe_model_card:
        model_card_id = 'test-model-card'
        expected_result = {
            'ModelCardId': model_card_id,
            'ModelCardArn': f'arn:aws:sagemaker:us-west-2:123456789012:model-card/{model_card_id}',
            'CreationTime': '2023-01-01T00:00:00',
        }
        mock_describe_model_card.return_value = expected_result

        result = await describe_model_card_sagemaker(model_card_id)

        mock_describe_model_card.assert_called_once_with(model_card_id)
        assert result == {'model_card_details': expected_result}


@pytest.mark.asyncio
async def test_list_apps_sagemaker():
    """Test list_apps_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.list_apps') as mock_list_apps:
        expected_result = [
            {
                'AppName': 'test-app-1',
                'AppType': 'JupyterServer',
                'DomainId': 'test-domain',
                'UserProfileName': 'test-user',
            },
            {
                'AppName': 'test-app-2',
                'AppType': 'KernelGateway',
                'DomainId': 'test-domain',
                'UserProfileName': 'test-user',
            },
        ]
        mock_list_apps.return_value = expected_result

        result = await list_apps_sagemaker()

        mock_list_apps.assert_called_once()
        assert result == {'apps': expected_result}


@pytest.mark.asyncio
async def test_create_app_sagemaker():
    """Test create_app_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.create_app') as mock_create_app:
        app_arn = 'arn:aws:sagemaker:us-west-2:123456789012:app/domain/user/app'
        mock_create_app.return_value = app_arn

        domain_id = 'test-domain'
        user_profile_name = 'test-user'
        app_type = 'JupyterServer'
        app_name = 'test-app'
        resource_spec = {'InstanceType': 'ml.t3.medium'}

        result = await create_app_sagemaker(
            domain_id=domain_id,
            user_profile_name=user_profile_name,
            app_type=app_type,
            app_name=app_name,
            resource_spec=resource_spec,
        )

        mock_create_app.assert_called_once_with(
            domain_id,
            user_profile_name,
            app_type,
            app_name,
            resource_spec,
        )
        assert result == {'app_arn': app_arn}


@pytest.mark.asyncio
async def test_create_presigned_notebook_instance_url_sagemaker():
    """Test create_presigned_notebook_instance_url_sagemaker function."""
    with patch(
        'sagemaker_ai_mcp_server.server.create_presigned_notebook_instance_url'
    ) as mock_create_url:
        notebook_name = 'test-notebook'
        expiration = 7200
        expected_url = 'https://example.com/presigned-notebook-url'
        mock_create_url.return_value = expected_url

        result = await create_presigned_notebook_instance_url_sagemaker(
            notebook_instance_name=notebook_name,
            session_expiration_duration_in_seconds=expiration,
        )

        mock_create_url.assert_called_once_with(notebook_name, expiration)
        assert result == {'presigned_url': expected_url}


@pytest.mark.asyncio
async def test_describe_app_sagemaker():
    """Test describe_app_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_app') as mock_describe_app:
        domain_id = 'test-domain'
        user_profile_name = 'test-user'
        app_type = 'JupyterServer'
        app_name = 'test-app'
        expected_result = {
            'AppName': app_name,
            'AppType': app_type,
            'DomainId': domain_id,
            'UserProfileName': user_profile_name,
            'Status': 'InService',
        }
        mock_describe_app.return_value = expected_result

        result = await describe_app_sagemaker(
            domain_id=domain_id,
            user_profile_name=user_profile_name,
            app_type=app_type,
            app_name=app_name,
        )

        mock_describe_app.assert_called_once_with(domain_id, user_profile_name, app_type, app_name)
        assert result == {'app_details': expected_result}


@pytest.mark.asyncio
async def test_describe_app_image_config_sagemaker():
    """Test describe_app_image_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.describe_app_image_config') as mock_describe_config:
        config_name = 'test-app-image-config'
        expected_result = {
            'AppImageConfigName': config_name,
            'CreationTime': '2023-01-01T00:00:00Z',
        }
        mock_describe_config.return_value = expected_result

        result = await describe_app_image_config_sagemaker(app_image_config_name=config_name)

        mock_describe_config.assert_called_once_with(config_name)
        assert result == {'app_image_config_details': expected_result}


@pytest.mark.asyncio
async def test_delete_app_sagemaker():
    """Test delete_app_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_app') as mock_delete_app:
        domain_id = 'test-domain'
        user_profile_name = 'test-user'
        app_type = 'JupyterServer'
        app_name = 'test-app'

        result = await delete_app_sagemaker(
            domain_id=domain_id,
            user_profile_name=user_profile_name,
            app_type=app_type,
            app_name=app_name,
        )

        mock_delete_app.assert_called_once_with(domain_id, user_profile_name, app_type, app_name)
        expected_msg = f"App '{app_name}' deletion initiated successfully"
        assert result == {'message': expected_msg}


@pytest.mark.asyncio
async def test_delete_app_image_config_sagemaker():
    """Test delete_app_image_config_sagemaker function."""
    with patch('sagemaker_ai_mcp_server.server.delete_app_image_config') as mock_delete_config:
        config_name = 'test-app-image-config'

        result = await delete_app_image_config_sagemaker(app_image_config_name=config_name)

        mock_delete_config.assert_called_once_with(config_name)
        expected_msg = f"App Image Config '{config_name}' deleted successfully"
        assert result == {'message': expected_msg}
