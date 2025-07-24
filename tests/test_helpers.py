"""Tests for the helper functions in the SageMaker AI MCP Server."""

import os
import pytest
from sagemaker_ai_mcp_server.helpers import (
    delete_endpoint,
    delete_endpoint_config,
    delete_pipeline,
    describe_endpoint,
    describe_endpoint_config,
    describe_pipeline,
    describe_pipeline_definition_for_execution,
    describe_pipeline_execution,
    describe_processing_job,
    describe_training_job,
    describe_transform_job,
    get_aws_session,
    get_region,
    get_sagemaker_client,
    list_endpoint_configs,
    list_endpoints,
    list_pipeline_execution_steps,
    list_pipeline_executions,
    list_pipeline_parameters_for_execution,
    list_pipelines,
    list_processing_jobs,
    list_training_jobs,
    list_transform_jobs,
    start_pipeline_execution,
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
    async def test_start_pipeline_execution_without_parameters(
        self, mock_get_sagemaker_client
    ):
        """Test start_pipeline_execution function without parameters."""
        mock_client = MagicMock()
        pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
        pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
        expected_response = {
            'PipelineExecutionArn': pipeline_arn + pipeline_path
        }
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
    async def test_start_pipeline_execution_with_parameters(
        self, mock_get_sagemaker_client
    ):
        """Test start_pipeline_execution function with pipeline parameters."""
        mock_client = MagicMock()
        pipeline_arn = 'arn:aws:sagemaker:us-west-2:123456789012:'
        pipeline_path = 'pipeline/test-pipeline/execution/test-execution'
        expected_response = {
            'PipelineExecutionArn': pipeline_arn + pipeline_path
        }
        mock_client.start_pipeline_execution.return_value = expected_response
        mock_get_sagemaker_client.return_value = mock_client

        pipeline_parameters = [
            {'Name': 'param1', 'Value': 'value1'},
            {'Name': 'param2', 'Value': 'value2'}
        ]

        response = await start_pipeline_execution(
            'test-pipeline', pipeline_parameters
        )

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
