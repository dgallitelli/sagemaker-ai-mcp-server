"""Tests for the helper functions in the SageMaker AI MCP Server."""

import os
import pytest
from sagemaker_ai_mcp_server.helpers import (
    create_mlflow_tracking_server,
    create_presigned_domain_url,
    create_presigned_mlflow_tracking_server_url,
    delete_domain,
    delete_endpoint,
    delete_endpoint_config,
    delete_mlflow_tracking_server,
    delete_model,
    delete_model_card,
    delete_pipeline,
    describe_domain,
    describe_endpoint,
    describe_endpoint_config,
    describe_mlflow_tracking_server,
    describe_model,
    describe_model_card,
    describe_pipeline,
    describe_pipeline_definition_for_execution,
    describe_pipeline_execution,
    describe_processing_job,
    describe_training_job,
    describe_transform_job,
    get_aws_session,
    get_region,
    get_sagemaker_client,
    get_sagemaker_execution_role_arn,
    list_domains,
    list_endpoint_configs,
    list_endpoints,
    list_mlflow_tracking_servers,
    list_model_card_export_jobs,
    list_model_card_versions,
    list_model_cards,
    list_models,
    list_pipeline_execution_steps,
    list_pipeline_executions,
    list_pipeline_parameters_for_execution,
    list_pipelines,
    list_processing_jobs,
    list_spaces,
    list_training_jobs,
    list_transform_jobs,
    list_user_profiles,
    start_mlflow_tracking_server,
    start_pipeline_execution,
    stop_mlflow_tracking_server,
    stop_pipeline_execution,
    stop_processing_job,
    stop_training_job,
    stop_transform_job,
)
from unittest.mock import MagicMock, patch


class TestHelpers:
    """Tests for the SageMaker AI MCP Server helper functions."""

    def test_get_region_with_env_var(self):
        """Test get_region with AWS_REGION environment variable set."""
        with patch.dict(os.environ, {'AWS_REGION': 'eu-west-1'}):
            assert get_region() == 'eu-west-1'

    def test_get_region_default(self):
        """Test get_region with no AWS_REGION environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_region() == 'us-east-1'

    def test_get_sagemaker_execution_role_arn(self):
        """Test get_sagemaker_execution_role_arn function."""
        with patch.dict(
            os.environ,
            {
                'SAGEMAKER_EXECUTION_ROLE_ARN': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
            },
        ):
            role_arn = get_sagemaker_execution_role_arn()
            assert role_arn == 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'

    @patch('sagemaker_ai_mcp_server.helpers.boto3.Session')
    def test_get_aws_session_with_profile(self, mock_session):
        """Test get_aws_session with AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {'AWS_PROFILE': 'test-profile'}):
            session = get_aws_session('eu-west-1')

        mock_session.assert_called_once_with(profile_name='test-profile', region_name='eu-west-1')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.boto3.Session')
    def test_get_aws_session_without_profile(self, mock_session):
        """Test get_aws_session without AWS_PROFILE environment variable."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch.dict(os.environ, {}, clear=True):
            session = get_aws_session('us-west-2')

        mock_session.assert_called_once_with(region_name='us-west-2')
        assert session == mock_session_instance

    @patch('sagemaker_ai_mcp_server.helpers.get_aws_session')
    def test_get_sagemaker_client(self, mock_get_aws_session):
        """Test get_sagemaker_client function."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_get_aws_session.return_value = mock_session

        client = get_sagemaker_client('us-west-1')

        mock_get_aws_session.assert_called_once_with('us-west-1')
        mock_session.client.assert_called_once_with('sagemaker')
        assert client == mock_client

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_endpoints(self, mock_get_sagemaker_client):
        """Test list_endpoints function."""
        mock_client = MagicMock()
        mock_client.list_endpoints.return_value = {
            'Endpoints': [{'EndpointName': 'test-endpoint'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        endpoints = await list_endpoints()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_endpoints.assert_called_once()
        assert endpoints == [{'EndpointName': 'test-endpoint'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_endpoint_configs(self, mock_get_sagemaker_client):
        """Test list_endpoint_configs function."""
        mock_client = MagicMock()
        mock_client.list_endpoint_configs.return_value = {
            'EndpointConfigs': [{'EndpointConfigName': 'test-config'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        configs = await list_endpoint_configs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_endpoint_configs.assert_called_once()
        assert configs == [{'EndpointConfigName': 'test-config'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_endpoint(self, mock_get_sagemaker_client):
        """Test delete_endpoint function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_endpoint('test-endpoint')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_endpoint.assert_called_once_with(EndpointName='test-endpoint')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_endpoint_config(self, mock_get_sagemaker_client):
        """Test delete_endpoint_config function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_endpoint_config('test-config')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_endpoint_config.assert_called_once_with(
            EndpointConfigName='test-config'
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_endpoint(self, mock_get_sagemaker_client):
        """Test describe_endpoint function."""
        mock_client = MagicMock()
        expected_response = {'EndpointName': 'test-endpoint', 'Status': 'InService'}
        mock_client.describe_endpoint.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_endpoint('test-endpoint')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_endpoint.assert_called_once_with(EndpointName='test-endpoint')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_endpoint_config(self, mock_get_sagemaker_client):
        """Test describe_endpoint_config function."""
        mock_client = MagicMock()
        expected_response = {'EndpointConfigName': 'test-config', 'ProductionVariants': []}
        mock_client.describe_endpoint_config.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_endpoint_config('test-config')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_endpoint_config.assert_called_once_with(
            EndpointConfigName='test-config'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_training_job(self, mock_get_sagemaker_client):
        """Test describe_training_job function."""
        mock_client = MagicMock()
        expected_response = {'TrainingJobName': 'test-job', 'TrainingJobStatus': 'Completed'}
        mock_client.describe_training_job.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_training_job('test-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_training_job.assert_called_once_with(TrainingJobName='test-job')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_training_jobs(self, mock_get_sagemaker_client):
        """Test list_training_jobs function."""
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {
            'TrainingJobSummaries': [{'TrainingJobName': 'test-job'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        jobs = await list_training_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_training_jobs.assert_called_once()
        assert jobs == [{'TrainingJobName': 'test-job'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_training_job(self, mock_get_sagemaker_client):
        """Test stop_training_job function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await stop_training_job('test-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_training_job.assert_called_once_with(TrainingJobName='test-job')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_processing_job(self, mock_get_sagemaker_client):
        """Test describe_processing_job function."""
        mock_client = MagicMock()
        expected_response = {
            'ProcessingJobName': 'test-processing-job',
            'ProcessingJobStatus': 'Completed',
        }
        mock_client.describe_processing_job.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_processing_job('test-processing-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_processing_job.assert_called_once_with(
            ProcessingJobName='test-processing-job'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_processing_jobs(self, mock_get_sagemaker_client):
        """Test list_processing_jobs function."""
        mock_client = MagicMock()
        mock_client.list_processing_jobs.return_value = {
            'ProcessingJobSummaries': [{'ProcessingJobName': 'test-processing-job'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        jobs = await list_processing_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_processing_jobs.assert_called_once()
        assert jobs == [{'ProcessingJobName': 'test-processing-job'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_processing_job(self, mock_get_sagemaker_client):
        """Test stop_processing_job function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await stop_processing_job('test-processing-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_processing_job.assert_called_once_with(
            ProcessingJobName='test-processing-job'
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_transform_job(self, mock_get_sagemaker_client):
        """Test describe_transform_job function."""
        mock_client = MagicMock()
        expected_response = {
            'TransformJobName': 'test-transform-job',
            'TransformJobStatus': 'Completed',
        }
        mock_client.describe_transform_job.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_transform_job('test-transform-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_transform_job.assert_called_once_with(
            TransformJobName='test-transform-job'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_transform_jobs(self, mock_get_sagemaker_client):
        """Test list_transform_jobs function."""
        mock_client = MagicMock()
        mock_client.list_transform_jobs.return_value = {
            'TransformJobSummaries': [{'TransformJobName': 'test-transform-job'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        jobs = await list_transform_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_transform_jobs.assert_called_once()
        assert jobs == [{'TransformJobName': 'test-transform-job'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_transform_job(self, mock_get_sagemaker_client):
        """Test stop_transform_job function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await stop_transform_job('test-transform-job')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_transform_job.assert_called_once_with(
            TransformJobName='test-transform-job'
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_pipelines(self, mock_get_sagemaker_client):
        """Test list_pipelines function."""
        mock_client = MagicMock()
        mock_client.list_pipelines.return_value = {
            'PipelineSummaries': [{'PipelineName': 'test-pipeline'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        pipelines = await list_pipelines()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_pipelines.assert_called_once()
        assert pipelines == [{'PipelineName': 'test-pipeline'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_pipeline(self, mock_get_sagemaker_client):
        """Test describe_pipeline function."""
        mock_client = MagicMock()
        expected_response = {'PipelineName': 'test-pipeline', 'PipelineStatus': 'Active'}
        mock_client.describe_pipeline.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_pipeline('test-pipeline')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_pipeline.assert_called_once_with(PipelineName='test-pipeline')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_pipeline_executions(self, mock_get_sagemaker_client):
        """Test list_pipeline_executions function."""
        mock_client = MagicMock()
        mock_client.list_pipeline_executions.return_value = {
            'PipelineExecutionSummaries': [
                {
                    'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
                }
            ]
        }
        mock_get_sagemaker_client.return_value = mock_client

        executions = await list_pipeline_executions('test-pipeline')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_pipeline_executions.assert_called_once_with(PipelineName='test-pipeline')
        assert executions == [
            {
                'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution'
            }
        ]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_pipeline_execution_steps(self, mock_get_sagemaker_client):
        """Test list_pipeline_execution_steps function."""
        mock_client = MagicMock()
        mock_client.list_pipeline_execution_steps.return_value = {
            'PipelineExecutionSteps': [{'StepName': 'test-step', 'StepStatus': 'Succeeded'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        steps = await list_pipeline_execution_steps('test-execution')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_pipeline_execution_steps.assert_called_once_with(
            PipelineExecutionArn='test-execution'
        )
        assert steps == [{'StepName': 'test-step', 'StepStatus': 'Succeeded'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_pipeline_parameters_for_execution(self, mock_get_sagemaker_client):
        """Test list_pipeline_parameters_for_execution function."""
        mock_client = MagicMock()
        mock_client.list_pipeline_parameters_for_execution.return_value = {
            'PipelineParameters': [{'Name': 'param1', 'Value': 'value1'}]
        }
        mock_get_sagemaker_client.return_value = mock_client

        parameters = await list_pipeline_parameters_for_execution('test-execution')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_pipeline_parameters_for_execution.assert_called_once_with(
            PipelineExecutionArn='test-execution'
        )
        assert parameters == [{'Name': 'param1', 'Value': 'value1'}]

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_pipeline(self, mock_get_sagemaker_client):
        """Test delete_pipeline function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_pipeline('test-pipeline')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_pipeline.assert_called_once_with(PipelineName='test-pipeline')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_pipeline_definition_for_execution(self, mock_get_sagemaker_client):
        """Test describe_pipeline_definition_for_execution function."""
        mock_client = MagicMock()
        expected_response = {
            'PipelineDefinition': 'pipeline-definition-content',
            'PipelineDefinitionS3Location': {'Bucket': 'test-bucket', 'Key': 'test-key'},
        }
        mock_client.describe_pipeline_definition_for_execution.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_pipeline_definition_for_execution('test-execution')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_pipeline_definition_for_execution.assert_called_once_with(
            PipelineExecutionArn='test-execution'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_pipeline_execution(self, mock_get_sagemaker_client):
        """Test describe_pipeline_execution function."""
        mock_client = MagicMock()
        expected_response = {
            'PipelineExecutionArn': 'arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline/execution/test-execution',
            'PipelineExecutionStatus': 'InProgress',
        }
        mock_client.describe_pipeline_execution.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await describe_pipeline_execution('test-execution')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_pipeline_execution.assert_called_once_with(
            PipelineExecutionArn='test-execution'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_start_pipeline_execution_without_parameters(self, mock_get_sagemaker_client):
        """Test start_pipeline_execution function without parameters."""
        mock_client = MagicMock()
        pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
        pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
        expected_response = {'PipelineExecutionArn': pipeline_arn + pipeline_path}
        mock_client.start_pipeline_execution.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        response = await start_pipeline_execution('test-pipeline')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.start_pipeline_execution.assert_called_once_with(
            PipelineName='test-pipeline',
            PipelineParameters=[],
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_start_pipeline_execution_with_parameters(self, mock_get_sagemaker_client):
        """Test start_pipeline_execution function with pipeline parameters."""
        mock_client = MagicMock()
        pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
        pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
        expected_response = {'PipelineExecutionArn': pipeline_arn + pipeline_path}
        mock_client.start_pipeline_execution.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        pipeline_parameters = [
            {'Name': 'param1', 'Value': 'value1'},
            {'Name': 'param2', 'Value': 'value2'},
        ]

        response = await start_pipeline_execution('test-pipeline', pipeline_parameters)

        mock_get_sagemaker_client.assert_called_once()
        mock_client.start_pipeline_execution.assert_called_once_with(
            PipelineName='test-pipeline',
            PipelineParameters=pipeline_parameters,
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_pipeline_execution(self, mock_get_sagemaker_client):
        """Test stop_pipeline_execution function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
        pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
        execution_arn = pipeline_arn + pipeline_path

        await stop_pipeline_execution(execution_arn)

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_pipeline_execution.assert_called_once_with(
            PipelineExecutionArn=execution_arn
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_execution_role_arn')
    async def test_create_mlflow_tracking_server(
        self, mock_get_role_arn, mock_get_sagemaker_client
    ):
        """Test create_mlflow_tracking_server function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client
        role_arn = 'arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole'
        mock_get_role_arn.return_value = role_arn

        await create_mlflow_tracking_server(
            'test-mlflow-server', 's3://bucket/artifacts', 'Medium'
        )

        mock_get_sagemaker_client.assert_called_once()
        mock_get_role_arn.assert_called_once()
        mock_client.create_mlflow_tracking_server.assert_called_once_with(
            TrackingServerName='test-mlflow-server',
            ArtifactStoreUri='s3://bucket/artifacts',
            TrackingServerSize='Medium',
            RoleArn=role_arn,
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_mlflow_tracking_server(self, mock_get_sagemaker_client):
        """Test delete_mlflow_tracking_server function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_mlflow_tracking_server('test-mlflow-server')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_mlflow_tracking_server.assert_called_once_with(
            TrackingServerName='test-mlflow-server'
        )

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_mlflow_tracking_servers(self, mock_get_sagemaker_client):
        """Test list_mlflow_tracking_servers function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'TrackingServerSummaries': [
                {'TrackingServerName': 'test-mlflow-server', 'Status': 'InService'}
            ]
        }
        mock_client.list_mlflow_tracking_servers.return_value = mock_response

        servers = await list_mlflow_tracking_servers()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_mlflow_tracking_servers.assert_called_once()
        expected = [{'TrackingServerName': 'test-mlflow-server', 'Status': 'InService'}]
        assert servers == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_mlflow_tracking_server(self, mock_get_sagemaker_client):
        """Test describe_mlflow_tracking_server function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {
            'TrackingServerName': 'test-mlflow-server',
            'Status': 'InService',
            'CreationTime': '2023-01-01T00:00:00Z',
        }
        mock_client.describe_mlflow_tracking_server.return_value = expected_response

        response = await describe_mlflow_tracking_server('test-mlflow-server')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_mlflow_tracking_server.assert_called_once_with(
            TrackingServerName='test-mlflow-server'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_start_mlflow_tracking_server(self, mock_get_sagemaker_client):
        """Test start_mlflow_tracking_server function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {'TrackingServerName': 'test-mlflow-server', 'Status': 'Starting'}
        mock_client.start_mlflow_tracking_server.return_value = expected_response

        response = await start_mlflow_tracking_server('test-mlflow-server')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.start_mlflow_tracking_server.assert_called_once_with(
            TrackingServerName='test-mlflow-server'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_stop_mlflow_tracking_server(self, mock_get_sagemaker_client):
        """Test stop_mlflow_tracking_server function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {'TrackingServerName': 'test-mlflow-server', 'Status': 'Stopping'}
        mock_client.stop_mlflow_tracking_server.return_value = expected_response

        response = await stop_mlflow_tracking_server('test-mlflow-server')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.stop_mlflow_tracking_server.assert_called_once_with(
            TrackingServerName='test-mlflow-server'
        )
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_create_presigned_mlflow_tracking_server_url_default(
        self, mock_get_sagemaker_client
    ):
        """Test create_presigned_mlflow_tracking_server_url with default expiration."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {'PresignedUrl': 'https://example.com/presigned-url'}
        mock_client.create_presigned_mlflow_tracking_server_url.return_value = expected_response

        url = await create_presigned_mlflow_tracking_server_url('test-mlflow-server')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.create_presigned_mlflow_tracking_server_url.assert_called_once_with(
            TrackingServerName='test-mlflow-server', ExpirationSeconds=3600
        )
        assert url == 'https://example.com/presigned-url'

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_create_presigned_mlflow_tracking_server_url_custom(
        self, mock_get_sagemaker_client
    ):
        """Test create_presigned_mlflow_tracking_server_url with custom expiration."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {'PresignedUrl': 'https://example.com/presigned-url-custom'}
        mock_client.create_presigned_mlflow_tracking_server_url.return_value = expected_response

        custom_expiration = 7200
        url = await create_presigned_mlflow_tracking_server_url(
            'test-mlflow-server', custom_expiration
        )

        mock_get_sagemaker_client.assert_called_once()
        mock_client.create_presigned_mlflow_tracking_server_url.assert_called_once_with(
            TrackingServerName='test-mlflow-server', ExpirationSeconds=custom_expiration
        )
        assert url == 'https://example.com/presigned-url-custom'

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_domain(self, mock_get_sagemaker_client):
        """Test delete_domain function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_domain('test-domain')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_domain.assert_called_once_with(DomainId='test-domain')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_domains(self, mock_get_sagemaker_client):
        """Test list_domains function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {'Domains': [{'DomainId': 'test-domain', 'DomainName': 'Test Domain'}]}
        mock_client.list_domains.return_value = mock_response

        domains = await list_domains()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_domains.assert_called_once()
        expected = [{'DomainId': 'test-domain', 'DomainName': 'Test Domain'}]
        assert domains == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_domain(self, mock_get_sagemaker_client):
        """Test describe_domain function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {
            'DomainId': 'test-domain',
            'DomainName': 'Test Domain',
            'Status': 'InService',
        }
        mock_client.describe_domain.return_value = expected_response

        response = await describe_domain('test-domain')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_domain.assert_called_once_with(DomainId='test-domain')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_create_presigned_domain_url(self, mock_get_sagemaker_client):
        """Test create_presigned_domain_url function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {'AuthorizedUrl': 'https://example.com/presigned-domain-url'}
        mock_client.create_presigned_domain_url.return_value = expected_response

        url = await create_presigned_domain_url('test-domain', 'test-profile-name')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.create_presigned_domain_url.assert_called_once_with(
            DomainId='test-domain', UserProfileName='test-profile-name', ExpirationSeconds=3600
        )
        assert url == 'https://example.com/presigned-domain-url'

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_spaces(self, mock_get_sagemaker_client):
        """Test list_spaces function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {'Spaces': [{'SpaceName': 'test-space', 'SpaceId': 'space-id-123'}]}
        mock_client.list_spaces.return_value = mock_response

        spaces = await list_spaces()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_spaces.assert_called_once()
        expected = [{'SpaceName': 'test-space', 'SpaceId': 'space-id-123'}]
        assert spaces == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_user_profiles(self, mock_get_sagemaker_client):
        """Test list_user_profiles function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'UserProfiles': [{'UserProfileName': 'test-user', 'UserProfileArn': 'arn:aws:...'}]
        }
        mock_client.list_user_profiles.return_value = mock_response

        profiles = await list_user_profiles()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_user_profiles.assert_called_once()
        expected = [{'UserProfileName': 'test-user', 'UserProfileArn': 'arn:aws:...'}]
        assert profiles == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_model(self, mock_get_sagemaker_client):
        """Test describe_model function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {
            'ModelName': 'test-model',
            'PrimaryContainer': {
                'Image': '123456789012.dkr.ecr.us-west-2.amazonaws.com/test-image:latest'
            },
            'ExecutionRoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
        }
        mock_client.describe_model.return_value = expected_response

        response = await describe_model('test-model')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_model.assert_called_once_with(ModelName='test-model')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_models(self, mock_get_sagemaker_client):
        """Test list_models function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'Models': [{'ModelName': 'test-model', 'CreationTime': '2023-01-01T00:00:00Z'}]
        }
        mock_client.list_models.return_value = mock_response

        models = await list_models()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_models.assert_called_once()
        expected = [{'ModelName': 'test-model', 'CreationTime': '2023-01-01T00:00:00Z'}]
        assert models == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_delete_model(self, mock_get_sagemaker_client):
        """Test delete_model function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_model('test-model')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_model.assert_called_once_with(ModelName='test-model')

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_model_cards(self, mock_get_sagemaker_client):
        """Test list_model_cards function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'ModelCardSummaries': [{'ModelCardName': 'test-card', 'ModelCardArn': 'arn:aws:...'}]
        }
        mock_client.list_model_cards.return_value = mock_response

        cards = await list_model_cards()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_model_cards.assert_called_once()
        expected = [{'ModelCardName': 'test-card', 'ModelCardArn': 'arn:aws:...'}]
        assert cards == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_describe_model_card(self, mock_get_sagemaker_client):
        """Test describe_model_card function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        expected_response = {
            'ModelCardName': 'test-card',
            'ModelCardArn': 'arn:aws:sagemaker:us-west-2:123456789012:model-card/test-card',
            'ModelCardStatus': 'Draft',
        }
        mock_client.describe_model_card.return_value = expected_response

        response = await describe_model_card('test-card')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.describe_model_card.assert_called_once_with(ModelCardName='test-card')
        assert response == expected_response

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_model_card_export_jobs(self, mock_get_sagemaker_client):
        """Test list_model_card_export_jobs function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'ModelCardExportJobSummaries': [
                {'ModelCardExportJobName': 'test-export-job', 'ModelCardArn': 'arn:aws:...'}
            ]
        }
        mock_client.list_model_card_export_jobs.return_value = mock_response

        jobs = await list_model_card_export_jobs()

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_model_card_export_jobs.assert_called_once()
        expected = [{'ModelCardExportJobName': 'test-export-job', 'ModelCardArn': 'arn:aws:...'}]
        assert jobs == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def test_list_model_card_versions(self, mock_get_sagemaker_client):
        """Test list_model_card_versions function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        mock_response = {
            'ModelCardVersionSummaryList': [
                {'ModelCardVersion': '1.0', 'ModelCardArn': 'arn:aws:...'}
            ]
        }
        mock_client.list_model_card_versions.return_value = mock_response

        versions = await list_model_card_versions('test-card')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.list_model_card_versions.assert_called_once_with(ModelCardName='test-card')
        expected = [{'ModelCardVersion': '1.0', 'ModelCardArn': 'arn:aws:...'}]
        assert versions == expected

    @pytest.mark.asyncio
    @patch('sagemaker_ai_mcp_server.helpers.get_sagemaker_client')
    async def delete_model_card(self, mock_get_sagemaker_client):
        """Test delete_model_card function."""
        mock_client = MagicMock()
        mock_get_sagemaker_client.return_value = mock_client

        await delete_model_card('test-card')

        mock_get_sagemaker_client.assert_called_once()
        mock_client.delete_model_card.assert_called_once_with(ModelCardName='test-card')