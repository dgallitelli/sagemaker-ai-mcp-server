"""The main file for the SageMaker AI MCP Server."""

from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
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
    describe_inference_recommendations_job,
    describe_mlflow_tracking_server,
    describe_model,
    describe_model_card,
    describe_pipeline,
    describe_pipeline_definition_for_execution,
    describe_pipeline_execution,
    describe_processing_job,
    describe_training_job,
    describe_transform_job,
    list_domains,
    list_endpoint_configs,
    list_endpoints,
    list_inference_recommendations_job_steps,
    list_inference_recommendations_jobs,
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
    stop_inference_recommendations_job,
    stop_mlflow_tracking_server,
    stop_pipeline_execution,
    stop_processing_job,
    stop_training_job,
    stop_transform_job,
)
from typing import Annotated, Any, Dict, List, Literal


mcp = FastMCP(
    'sagemaker-ai-mcp-server',
    instructions="""
    SageMaker AI MCP Server provides tools to interact with Amazon SageMaker AI
    service.
    The server enables you to:
    - List SageMaker Endpoints
    - List SageMaker Endpoint Configurations
    - Delete SageMaker Endpoints
    - Delete SageMaker Endpoint Configurations
    - Describe SageMaker Endpoints
    - Describe SageMaker Endpoint Configurations
    - List SageMaker Training Jobs
    - Describe SageMaker Training Jobs
    - Stop SageMaker Training Jobs
    - List SageMaker Processing Jobs
    - Describe SageMaker Processing Jobs
    - Stop SageMaker Processing Jobs
    - List SageMaker Transform Jobs
    - Describe SageMaker Transform Jobs
    - Stop SageMaker Transform Jobs
    - List SageMaker Pipelines
    - Describe SageMaker Pipelines
    - Delete SageMaker Pipelines
    - List Pipeline Executions
    - List Pipeline Execution Steps
    - List Pipeline Parameters for Execution
    - Describe Pipeline Definition for Execution
    - Describe Pipeline Execution
    - Start Pipeline Execution
    - Stop Pipeline Execution
    - Create an Managed MLflow Tracking Server in SageMaker
    - Delete an Managed MLflow Tracking Server in SageMaker
    - List Managed MLflow Tracking Servers in SageMaker
    - Describe an Managed MLflow Tracking Server in SageMaker
    - Start an Managed MLflow Tracking Server in SageMaker
    - Stop an Managed MLflow Tracking Server in SageMaker
    - Create a presigned URL for an Managed MLflow Tracking Server in SageMaker
    - Delete a SageMaker Domain
    - List SageMaker Domains
    - Describe a SageMaker Domain
    - Create a presigned URL for a SageMaker Domain
    - List SageMaker Spaces
    - List SageMaker User Profiles
    - Describe a Model in SageMaker
    - List Models in SageMaker
    - Delete a Model in SageMaker
    - Describe a Model Card in SageMaker
    - List Model Cards in SageMaker
    - Delete a Model Card in SageMaker
    - Describe a Model Card Export Job in SageMaker
    - List Model Card Export Jobs in SageMaker
    - List Model Card Versions in SageMaker
    - List Inference Recommender Jobs
    - List Inference Recommender Job Steps
    - Describe Inference Recommender Job
    - Stop Inference Recommender Job
    Use these tools to manage your SageMaker resources effectively.
    """,
    dependencies=[
        'pydantic',
        'loguru',
        'boto3',
    ],
)


@mcp.tool(name='list_endpoints_sagemaker', description='List all SageMaker Endpoints')
async def list_endpoints_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoints in your account
    in the current region. This is typically used first to see what endpoints
    are available before performing operations on them.

    ## Example

    ```python
    endpoints = await list_endpoints_sagemaker()
    print(endpoints)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoints': A list of dictionaries, each representing a SageMaker
      Endpoint with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoints.
    """
    try:
        endpoints = await list_endpoints()
        return {'endpoints': endpoints}
    except Exception as e:
        logger.error(f'Error listing SageMaker endpoints: {e}')
        raise ValueError(f'Failed to list endpoints: {e}')


@mcp.tool(
    name='list_endpoint_configs_sagemaker',
    description='List all SageMaker Endpoint Configurations',
)
async def list_endpoint_configs_sagemaker() -> Dict[str, List]:
    """Get a list of all SageMaker Endpoint Configurations.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Endpoint Configurations
    in your account in the current region. This helps you identify available
    endpoint configurations before performing operations on them.

    ## Example

    ```python
    configs = await list_endpoint_configs_sagemaker()
    print(configs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'endpoint_configs': A list of dictionaries, each representing a SageMaker
      Endpoint Configuration with its details.

    ## Returns
    A dictionary containing a list of SageMaker Endpoint Configurations.
    """
    try:
        configs = await list_endpoint_configs()
        return {'endpoint_configs': configs}
    except Exception as e:
        logger.error(f'Error listing endpoint configurations: {e}')
        raise ValueError(f'Failed to list endpoint configs: {e}')


@mcp.tool(name='delete_endpoint_sagemaker', description='Delete a SageMaker Endpoint')
async def delete_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint.

    ## Usage

    Use this tool to delete a SageMaker Endpoint by providing its name.
    This is useful for cleaning up resources that are no longer needed.

    ## Example

    ```python
    result = await delete_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint(endpoint_name)
        return {'message': f"Endpoint '{endpoint_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to delete endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='delete_endpoint_config_sagemaker',
    description='Delete a SageMaker Endpoint Configuration',
)
async def delete_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to delete a SageMaker Endpoint Configuration by providing
    its name. This is useful for cleaning up configurations that are no longer
    needed.

    ## Example

    ```python
    result = await delete_endpoint_config_sagemaker(endpoint_config_name='my-endpoint-config')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_endpoint_config(endpoint_config_name)
        msg = f"Endpoint Config '{endpoint_config_name}' deleted successfully"
        return {'message': msg}
    except Exception as e:
        logger.error(f'Error deleting config {endpoint_config_name}: {e}')
        raise ValueError(f'Failed to delete config {endpoint_config_name}: {e}')


@mcp.tool(name='describe_endpoint_sagemaker', description='Describe a SageMaker Endpoint')
async def describe_endpoint_sagemaker(
    endpoint_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint by
    providing its name. This returns comprehensive information about the
    endpoint's configuration, status, and other details.

    ## Example

    ```python
    details = await describe_endpoint_sagemaker(endpoint_name='my-endpoint')
    print(details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint.

    ## Returns
    A dictionary containing the endpoint details.
    """
    try:
        endpoint_details = await describe_endpoint(endpoint_name)
        return {'endpoint_details': endpoint_details}
    except Exception as e:
        logger.error(f'Error describing endpoint {endpoint_name}: {e}')
        raise ValueError(f'Failed to describe endpoint {endpoint_name}: {e}')


@mcp.tool(
    name='describe_endpoint_config_sagemaker',
    description='Describe a SageMaker Endpoint Configuration',
)
async def describe_endpoint_config_sagemaker(
    endpoint_config_name: Annotated[
        str, Field(description='The name of the SageMaker Endpoint Configuration to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Endpoint Configuration.

    ## Usage

    Use this tool to get detailed information about a SageMaker Endpoint
    Configuration by providing its name. This returns comprehensive information
    about the configuration details, including model specifications, instance
    types, and other configuration parameters.

    ## Example

    ```python
    config_details = await describe_endpoint_config_sagemaker(
        endpoint_config_name='my-endpoint-config'
    )
    print(config_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Endpoint Configuration.

    ## Returns
    A dictionary containing the endpoint configuration details.
    """
    try:
        config_details = await describe_endpoint_config(endpoint_config_name)
        return {'endpoint_config_details': config_details}
    except Exception as e:
        logger.error(f'Error describing config {endpoint_config_name}: {e}')
        err_msg = f'Failed to describe config {endpoint_config_name}: {e}'
        raise ValueError(err_msg)


@mcp.tool(name='list_training_jobs_sagemaker', description='List SageMaker Training Jobs')
async def list_training_jobs_sagemaker() -> Dict[str, List]:
    """List all SageMaker Training Jobs.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Training Jobs in your
    account in the current region. This is typically used to see what training
    jobs are available before performing operations on them.

    ## Example

    ```python
    jobs = await list_training_jobs_sagemaker()
    print(jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'training_jobs': A list of dictionaries, each representing a SageMaker
      Training Job with its details.

    ## Returns
    A dictionary containing a list of SageMaker Training Jobs.
    """
    try:
        jobs = await list_training_jobs()
        return {'training_jobs': jobs}
    except Exception as e:
        logger.error(f'Error listing training jobs: {e}')
        raise ValueError(f'Failed to list training jobs: {e}')


@mcp.tool(name='describe_training_job_sagemaker', description='Describe a SageMaker Training Job')
async def describe_training_job_sagemaker(
    training_job_name: Annotated[
        str, Field(description='The name of the SageMaker Training Job to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Training Job.

    ## Usage

    Use this tool to get detailed information about a SageMaker Training Job
    by providing its name. This returns comprehensive information about the
    job's configuration, status, and other details.

    ## Example

    ```python
    job_details = await describe_training_job_sagemaker(training_job_name='my-training-job')
    print(job_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Training Job.

    ## Returns
    A dictionary containing the training job details.
    """
    try:
        job_details = await describe_training_job(training_job_name)
        return {'training_job_details': job_details}
    except Exception as e:
        logger.error(f'Error describing training job {training_job_name}: {e}')
        raise ValueError(f'Failed to describe training job {training_job_name}: {e}')


@mcp.tool(name='stop_training_job_sagemaker', description='Stop a SageMaker Training Job')
async def stop_training_job_sagemaker(
    training_job_name: Annotated[
        str, Field(description='The name of the SageMaker Training Job to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Training Job.

    ## Usage

    Use this tool to stop a SageMaker Training Job by providing its name.
    This is useful for terminating jobs that are no longer needed or are taking
    too long to complete.

    ## Example

    ```python
    result = await stop_training_job_sagemaker(training_job_name='my-training-job')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_training_job(training_job_name)
        return {'message': f"Training Job '{training_job_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping training job {training_job_name}: {e}')
        raise ValueError(f'Failed to stop training job {training_job_name}: {e}')


@mcp.tool(name='list_processing_jobs_sagemaker', description='List SageMaker Processing Jobs')
async def list_processing_jobs_sagemaker() -> Dict[str, List]:
    """List all SageMaker Processing Jobs.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Processing Jobs in your
    account in the current region. This is typically used to see what processing
    jobs are available before performing operations on them.

    ## Example

    ```python
    jobs = await list_processing_jobs_sagemaker()
    print(jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'processing_jobs': A list of dictionaries, each representing a SageMaker
      Processing Job with its details.

    ## Returns
    A dictionary containing a list of SageMaker Processing Jobs.
    """
    try:
        jobs = await list_processing_jobs()
        return {'processing_jobs': jobs}
    except Exception as e:
        logger.error(f'Error listing processing jobs: {e}')
        raise ValueError(f'Failed to list processing jobs: {e}')


@mcp.tool(
    name='describe_processing_job_sagemaker', description='Describe a SageMaker Processing Job'
)
async def describe_processing_job_sagemaker(
    processing_job_name: Annotated[
        str, Field(description='The name of the SageMaker Processing Job to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Processing Job.

    ## Usage

    Use this tool to get detailed information about a SageMaker Processing Job
    by providing its name. This returns comprehensive information about the
    job's configuration, status, and other details.

    ## Example

    ```python
    job_details = await describe_processing_job_sagemaker(processing_job_name='my-processing-job')
    print(job_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Processing Job.

    ## Returns
    A dictionary containing the processing job details.
    """
    try:
        job_details = await describe_processing_job(processing_job_name)
        return {'processing_job_details': job_details}
    except Exception as e:
        logger.error(f'Error describing processing job {processing_job_name}: {e}')
        raise ValueError(f'Failed to describe processing job {processing_job_name}: {e}')


@mcp.tool(name='stop_processing_job_sagemaker', description='Stop a SageMaker Processing Job')
async def stop_processing_job_sagemaker(
    processing_job_name: Annotated[
        str, Field(description='The name of the SageMaker Processing Job to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Processing Job.

    ## Usage

    Use this tool to stop a SageMaker Processing Job by providing its name.
    This is useful for terminating jobs that are no longer needed or are taking
    too long to complete.

    ## Example

    ```python
    result = await stop_processing_job_sagemaker(processing_job_name='my-processing-job')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_processing_job(processing_job_name)
        return {'message': f"Processing Job '{processing_job_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping processing job {processing_job_name}: {e}')
        raise ValueError(f'Failed to stop processing job {processing_job_name}: {e}')


@mcp.tool(name='list_transform_jobs_sagemaker', description='List SageMaker Transform Jobs')
async def list_transform_jobs_sagemaker() -> Dict[str, List]:
    """List all SageMaker Transform Jobs.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Transform Jobs in your
    account in the current region. This is typically used to see what transform
    jobs are available before performing operations on them.

    ## Example

    ```python
    jobs = await list_transform_jobs_sagemaker()
    print(jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'transform_jobs': A list of dictionaries, each representing a SageMaker
      Transform Job with its details.

    ## Returns
    A dictionary containing a list of SageMaker Transform Jobs.
    """
    try:
        jobs = await list_transform_jobs()
        return {'transform_jobs': jobs}
    except Exception as e:
        logger.error(f'Error listing transform jobs: {e}')
        raise ValueError(f'Failed to list transform jobs: {e}')


@mcp.tool(
    name='describe_transform_job_sagemaker', description='Describe a SageMaker Transform Job'
)
async def describe_transform_job_sagemaker(
    transform_job_name: Annotated[
        str, Field(description='The name of the SageMaker Transform Job to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Transform Job.

    ## Usage

    Use this tool to get detailed information about a SageMaker Transform Job
    by providing its name. This returns comprehensive information about the
    job's configuration, status, and other details.

    ## Example

    ```python
    job_details = await describe_transform_job_sagemaker(transform_job_name='my-transform-job')
    print(job_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Transform Job.

    ## Returns
    A dictionary containing the transform job details.
    """
    try:
        job_details = await describe_transform_job(transform_job_name)
        return {'transform_job_details': job_details}
    except Exception as e:
        logger.error(f'Error describing transform job {transform_job_name}: {e}')
        raise ValueError(f'Failed to describe transform job {transform_job_name}: {e}')


@mcp.tool(name='stop_transform_job_sagemaker', description='Stop a SageMaker Transform Job')
async def stop_transform_job_sagemaker(
    transform_job_name: Annotated[
        str, Field(description='The name of the SageMaker Transform Job to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Transform Job.

    ## Usage

    Use this tool to stop a SageMaker Transform Job by providing its name.
    This is useful for terminating jobs that are no longer needed or are taking
    too long to complete.

    ## Example

    ```python
    result = await stop_transform_job_sagemaker(transform_job_name='my-transform-job')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_transform_job(transform_job_name)
        return {'message': f"Transform Job '{transform_job_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping transform job {transform_job_name}: {e}')
        raise ValueError(f'Failed to stop transform job {transform_job_name}: {e}')


@mcp.tool(name='list_pipelines_sagemaker', description='List SageMaker Pipelines')
async def list_pipelines_sagemaker() -> Dict[str, List]:
    """List all SageMaker Pipelines.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Pipelines in your account
    in the current region. This is typically used to see what pipelines are
    available before performing operations on them.

    ## Example

    ```python
    pipelines = await list_pipelines_sagemaker()
    print(pipelines)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'pipelines': A list of dictionaries, each representing a SageMaker
      Pipeline with its details.

    ## Returns
    A dictionary containing a list of SageMaker Pipelines.
    """
    try:
        pipelines = await list_pipelines()
        return {'pipelines': pipelines}
    except Exception as e:
        logger.error(f'Error listing pipelines: {e}')
        raise ValueError(f'Failed to list pipelines: {e}')


@mcp.tool(name='describe_pipeline_sagemaker', description='Describe a SageMaker Pipeline')
async def describe_pipeline_sagemaker(
    pipeline_name: Annotated[
        str, Field(description='The name of the SageMaker Pipeline to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Pipeline.

    ## Usage

    Use this tool to get detailed information about a SageMaker Pipeline by
    providing its name. This returns comprehensive information about the
    pipeline's configuration, status, and other details.

    ## Example

    ```python
    pipeline_details = await describe_pipeline_sagemaker(pipeline_name='my-pipeline')
    print(pipeline_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Pipeline.

    ## Returns
    A dictionary containing the pipeline details.
    """
    try:
        pipeline_details = await describe_pipeline(pipeline_name)
        return {'pipeline_details': pipeline_details}
    except Exception as e:
        logger.error(f'Error describing pipeline {pipeline_name}: {e}')
        raise ValueError(f'Failed to describe pipeline {pipeline_name}: {e}')


@mcp.tool(name='delete_pipeline_sagemaker', description='Delete a SageMaker Pipeline')
async def delete_pipeline_sagemaker(
    pipeline_name: Annotated[
        str, Field(description='The name of the SageMaker Pipeline to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Pipeline.

    ## Usage

    Use this tool to delete a SageMaker Pipeline by providing its name.
    This is useful for cleaning up pipelines that are no longer needed.

    ## Example

    ```python
    result = await delete_pipeline_sagemaker(pipeline_name='my-pipeline')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_pipeline(pipeline_name)
        return {'message': f"Pipeline '{pipeline_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting pipeline {pipeline_name}: {e}')
        raise ValueError(f'Failed to delete pipeline {pipeline_name}: {e}')


@mcp.tool(
    name='list_pipeline_executions_sagemaker',
    description='List all Pipeline Executions for a SageMaker Pipeline',
)
async def list_pipeline_executions_sagemaker(
    pipeline_name: Annotated[
        str, Field(description='The name of the SageMaker Pipeline to list executions for')
    ],
) -> Dict[str, List]:
    """List all Pipeline Executions for a specified SageMaker Pipeline.

    ## Usage

    Use this tool to retrieve a list of all executions for a specific SageMaker
    Pipeline by providing its name. This helps you track the execution history
    of the pipeline.

    ## Example

    ```python
    executions = await list_pipeline_executions_sagemaker(pipeline_name='my-pipeline')
    print(executions)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'pipeline_executions': A list of dictionaries, each representing a
      SageMaker Pipeline Execution with its details.

    ## Returns
    A dictionary containing a list of Pipeline Executions.
    """
    try:
        executions = await list_pipeline_executions(pipeline_name)
        return {'pipeline_executions': executions}
    except Exception as e:
        logger.error(f'Error listing pipeline executions for {pipeline_name}: {e}')
        raise ValueError(f'Failed to list pipeline executions for {pipeline_name}: {e}')


@mcp.tool(
    name='list_pipeline_execution_steps_sagemaker',
    description='List all Pipeline Execution Steps for a SageMaker Pipeline Execution',
)
async def list_pipeline_execution_steps_sagemaker(
    pipeline_execution_arn: Annotated[
        str, Field(description='The ARN of the SageMaker Pipeline Execution to list steps for')
    ],
) -> Dict[str, List]:
    """List all Pipeline Execution Steps for a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to retrieve a list of all steps for a specific SageMaker
    Pipeline Execution by providing its ARN. This helps you track the execution
    flow and status of each step in the pipeline.

    ## Example

    ```python
    steps = await list_pipeline_execution_steps_sagemaker(
        pipeline_execution_arn='arn:aws:sagemaker:...'
    )
    print(steps)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'pipeline_execution_steps': A list of dictionaries, each representing a
      step in the SageMaker Pipeline Execution with its details.

    ## Returns
    A dictionary containing a list of Pipeline Execution Steps.
    """
    try:
        steps = await list_pipeline_execution_steps(pipeline_execution_arn)
        return {'pipeline_execution_steps': steps}
    except Exception as e:
        logger.error(f'Error listing pipeline execution steps for {pipeline_execution_arn}: {e}')
        raise ValueError(
            f'Failed to list pipeline execution steps for {pipeline_execution_arn}: {e}'
        )


@mcp.tool(
    name='list_pipeline_parameters_for_execution_sagemaker',
    description='List Pipeline Parameters for a SageMaker Pipeline Execution',
)
async def list_pipeline_parameters_for_execution_sagemaker(
    pipeline_execution_arn: Annotated[
        str,
        Field(description='The ARN of the SageMaker Pipeline Execution to list parameters for'),
    ],
) -> Dict[str, List]:
    """List Pipeline Parameters for a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to retrieve a list of all parameters for a specific SageMaker
    Pipeline Execution by providing its ARN. This helps you understand the
    input parameters used in the execution.

    ## Example

    ```python
    parameters = await list_pipeline_parameters_for_execution_sagemaker(
        pipeline_execution_arn='arn:aws:sagemaker:...'
    )
    print(parameters)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'pipeline_parameters': A list of dictionaries, each representing a
      parameter in the SageMaker Pipeline Execution with its details.

    ## Returns
    A dictionary containing a list of Pipeline Parameters.
    """
    try:
        parameters = await list_pipeline_parameters_for_execution(pipeline_execution_arn)
        return {'pipeline_parameters': parameters}
    except Exception as e:
        logger.error(f'Error listing pipeline parameters for {pipeline_execution_arn}: {e}')
        raise ValueError(f'Failed to list pipeline parameters for {pipeline_execution_arn}: {e}')


@mcp.tool(
    name='describe_pipeline_definition_for_execution_sagemaker',
    description='Describe Pipeline Definition for a SageMaker Pipeline Execution',
)
async def describe_pipeline_definition_for_execution_sagemaker(
    pipeline_execution_arn: Annotated[
        str,
        Field(
            description='The ARN of the SageMaker Pipeline Execution to describe definition for'
        ),
    ],
) -> Dict[str, Any]:
    """Describe the Pipeline Definition for a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to retrieve the definition of a specific SageMaker Pipeline Execution by providing its ARN.
    This helps you understand the structure and components of the pipeline.

    ## Example

    ```python
    definition = await describe_pipeline_definition_for_execution_sagemaker(
        pipeline_execution_arn='arn:aws:sagemaker:...'
    )
    print(definition)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'pipeline_definition': A dictionary representing the definition of the SageMaker Pipeline Execution.

    ## Returns
    A dictionary containing the Pipeline Definition.
    """
    try:
        definition = await describe_pipeline_definition_for_execution(pipeline_execution_arn)
        return {'pipeline_definition': definition}
    except Exception as e:
        logger.error(f'Error describing pipeline definition for {pipeline_execution_arn}: {e}')
        raise ValueError(
            f'Failed to describe pipeline definition for {pipeline_execution_arn}: {e}'
        )


@mcp.tool(
    name='describe_pipeline_execution_sagemaker',
    description='Describe a SageMaker Pipeline Execution',
)
async def describe_pipeline_execution_sagemaker(
    pipeline_execution_arn: Annotated[
        str,
        Field(description='The ARN of the SageMaker Pipeline Execution to describe'),
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to get detailed information about a SageMaker Pipeline Execution
    by providing its ARN. This returns comprehensive information about the execution's
    status, parameters, and other details.

    ## Example

    ```python
    execution_details = await describe_pipeline_execution_sagemaker(
        pipeline_execution_arn='arn:aws:sagemaker:...'
    )
    print(execution_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker Pipeline Execution.

    ## Returns
    A dictionary containing the pipeline execution details.
    """
    try:
        execution_details = await describe_pipeline_execution(pipeline_execution_arn)
        return {'pipeline_execution_details': execution_details}
    except Exception as e:
        logger.error(f'Error describing pipeline execution {pipeline_execution_arn}: {e}')
        raise ValueError(f'Failed to describe pipeline execution {pipeline_execution_arn}: {e}')


@mcp.tool(
    name='start_pipeline_execution_sagemaker', description='Start a SageMaker Pipeline Execution'
)
async def start_pipeline_execution_sagemaker(
    pipeline_name: Annotated[
        str, Field(description='The name of the SageMaker Pipeline to start execution for')
    ],
    parameters: Annotated[
        Dict[str, Any],
        Field(description='A dictionary of parameters to pass to the pipeline execution'),
    ],
) -> Dict[str, str]:
    """Start a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to start a SageMaker Pipeline Execution by providing the pipeline name
    and parameters. This is useful for triggering the execution of a pipeline with specific inputs.

    ## Example

    ```python
    result = await start_pipeline_execution_sagemaker(
        pipeline_name='my-pipeline', parameters={'param1': 'value1', 'param2': 'value2'}
    )
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        execution_arn = await start_pipeline_execution(pipeline_name, parameters)
        return {
            'message': f"Pipeline '{pipeline_name}' started successfully with ARN: {execution_arn}"
        }
    except Exception as e:
        logger.error(f'Error starting pipeline execution for {pipeline_name}: {e}')
        raise ValueError(f'Failed to start pipeline execution for {pipeline_name}: {e}')


@mcp.tool(
    name='stop_pipeline_execution_sagemaker', description='Stop a SageMaker Pipeline Execution'
)
async def stop_pipeline_execution_sagemaker(
    pipeline_execution_arn: Annotated[
        str, Field(description='The ARN of the SageMaker Pipeline Execution to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Pipeline Execution.

    ## Usage

    Use this tool to stop a SageMaker Pipeline Execution by providing its ARN.
    This is useful for terminating executions that are no longer needed or are taking
    too long to complete.

    ## Example

    ```python
    result = await stop_pipeline_execution_sagemaker(
        pipeline_execution_arn='arn:aws:sagemaker:...'
    )
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_pipeline_execution(pipeline_execution_arn)
        return {'message': f"Pipeline Execution '{pipeline_execution_arn}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping pipeline execution {pipeline_execution_arn}: {e}')
        raise ValueError(f'Failed to stop pipeline execution {pipeline_execution_arn}: {e}')


@mcp.tool(
    name='create_mlflow_tracking_server_sagemaker',
    description='Create a Managed MLflow Tracking Server in SageMaker',
)
async def create_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str, Field(description='The name of the MLflow Tracking Server to create')
    ],
    artifact_store_uri: Annotated[
        str, Field(description='The S3 URI for the artifact store of the MLflow Tracking Server')
    ],
    tracking_server_size: Annotated[
        Literal['Small', 'Medium', 'Large'],
        Field(description='The size of the MLflow Tracking Server to create'),
    ],
) -> Dict[str, str]:
    """Create a Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to create a managed MLflow Tracking Server in SageMaker by providing
    the server name, artifact store URI, and server size. This is useful for setting up
    a centralized tracking server for ML experiments.

    ## Example

    ```python
    result = await create_mlflow_tracking_server_sagemaker(
        tracking_server_name='my-tracking-server',
        artifact_store_uri='s3://my-bucket/mlflow-artifacts',
        tracking_server_size='Medium',
    )
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await create_mlflow_tracking_server(
            tracking_server_name, artifact_store_uri, tracking_server_size
        )
        return {'message': f"MLflow Tracking Server '{tracking_server_name}' created successfully"}
    except Exception as e:
        logger.error(f'Error creating MLflow Tracking Server {tracking_server_name}: {e}')
        raise ValueError(f'Failed to create MLflow Tracking Server {tracking_server_name}: {e}')


@mcp.tool(
    name='delete_mlflow_tracking_server_sagemaker',
    description='Delete a Managed MLflow Tracking Server in SageMaker',
)
async def delete_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str, Field(description='The name of the MLflow Tracking Server to delete')
    ],
) -> Dict[str, str]:
    """Delete a Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to delete a managed MLflow Tracking Server in SageMaker by providing
    the server name. This is useful for cleaning up resources that are no longer needed.

    ## Example

    ```python
    result = await delete_mlflow_tracking_server_sagemaker(
        tracking_server_name='my-tracking-server'
    )
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_mlflow_tracking_server(tracking_server_name)
        return {'message': f"MLflow Tracking Server '{tracking_server_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting MLflow Tracking Server {tracking_server_name}: {e}')
        raise ValueError(f'Failed to delete MLflow Tracking Server {tracking_server_name}: {e}')


@mcp.tool(
    name='list_mlflow_tracking_servers_sagemaker',
    description='List all Managed MLflow Tracking Servers in SageMaker',
)
async def list_mlflow_tracking_servers_sagemaker() -> Dict[str, List]:
    """List all Managed MLflow Tracking Servers in SageMaker.

    ## Usage

    Use this tool to retrieve a list of all managed MLflow Tracking Servers in your
    SageMaker account. This is useful for seeing what tracking servers are available.

    ## Example

    ```python
    servers = await list_mlflow_tracking_servers_sagemaker()
    print(servers)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'tracking_servers': A list of dictionaries, each representing a managed MLflow Tracking Server.

    ## Returns
    A dictionary containing a list of MLflow Tracking Servers.
    """
    try:
        servers = await list_mlflow_tracking_servers()
        return {'tracking_servers': servers}
    except Exception as e:
        logger.error(f'Error listing MLflow Tracking Servers: {e}')
        raise ValueError(f'Failed to list MLflow Tracking Servers: {e}')


@mcp.tool(
    name='describe_mlflow_tracking_server_sagemaker',
    description='Describe a Managed MLflow Tracking Server in SageMaker',
)
async def describe_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str, Field(description='The name of the MLflow Tracking Server to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to get detailed information about a managed MLflow Tracking Server
    by providing its name. This returns comprehensive information about the server's
    configuration, status, and other details.

    ## Example

    ```python
    server_details = await describe_mlflow_tracking_server_sagemaker(
        tracking_server_name='my-tracking-server'
    )
    print(server_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the MLflow Tracking Server.

    ## Returns
    A dictionary containing the tracking server details.
    """
    try:
        server_details = await describe_mlflow_tracking_server(tracking_server_name)
        return {'tracking_server_details': server_details}
    except Exception as e:
        logger.error(f'Error describing MLflow Tracking Server {tracking_server_name}: {e}')
        raise ValueError(f'Failed to describe MLflow Tracking Server {tracking_server_name}: {e}')


@mcp.tool(
    name='create_presigned_url_for_mlflow_tracking_server_sagemaker',
    description='Create a presigned URL for a Managed MLflow Tracking Server in SageMaker',
)
async def create_presigned_url_for_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str,
        Field(description='The name of the MLflow Tracking Server to create a presigned URL for'),
    ],
    expiration_seconds: Annotated[
        int, Field(description='The number of seconds the presigned URL should be valid for')
    ],
) -> Dict[str, str]:
    """Create a presigned URL for a Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to create a presigned URL for accessing a managed MLflow Tracking Server
    by providing the server name and expiration time. This is useful for securely sharing
    access to the tracking server.

    ## Example

    ```python
    url = await create_presigned_url_for_mlflow_tracking_server_sagemaker(
        tracking_server_name='my-tracking-server', expiration_seconds=3600
    )
    print(url)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'presigned_url': The generated presigned URL for the MLflow Tracking Server.

    ## Returns
    A dictionary containing the presigned URL.
    """
    try:
        presigned_url = await create_presigned_mlflow_tracking_server_url(
            tracking_server_name, expiration_seconds
        )
        return {'presigned_url': presigned_url}
    except Exception as e:
        logger.error(
            f'Error creating presigned URL for MLflow Tracking Server {tracking_server_name}: {e}'
        )
        raise ValueError(
            f'Failed to create presigned URL for MLflow Tracking Server {tracking_server_name}: {e}'
        )


@mcp.tool(
    name='start_mlflow_tracking_server_sagemaker',
    description='Start a Managed MLflow Tracking Server in SageMaker',
)
async def start_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str, Field(description='The name of the MLflow Tracking Server to start')
    ],
) -> Dict[str, str]:
    """Start a Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to start a managed MLflow Tracking Server in SageMaker by providing
    the server name. This is useful for activating a tracking server that has been created.

    ## Example

    ```python
    result = await start_mlflow_tracking_server_sagemaker(
        tracking_server_name='my-tracking-server'
    )
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await start_mlflow_tracking_server(tracking_server_name)
        return {'message': f"MLflow Tracking Server '{tracking_server_name}' started successfully"}
    except Exception as e:
        logger.error(f'Error starting MLflow Tracking Server {tracking_server_name}: {e}')
        raise ValueError(f'Failed to start MLflow Tracking Server {tracking_server_name}: {e}')


@mcp.tool(
    name='stop_mlflow_tracking_server_sagemaker',
    description='Stop a Managed MLflow Tracking Server in SageMaker',
)
async def stop_mlflow_tracking_server_sagemaker(
    tracking_server_name: Annotated[
        str, Field(description='The name of the MLflow Tracking Server to stop')
    ],
) -> Dict[str, str]:
    """Stop a Managed MLflow Tracking Server in SageMaker.

    ## Usage

    Use this tool to stop a managed MLflow Tracking Server in SageMaker by providing
    the server name. This is useful for deactivating a tracking server that is no longer needed.

    ## Example

    ```python
    result = await stop_mlflow_tracking_server_sagemaker(tracking_server_name='my-tracking-server')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_mlflow_tracking_server(tracking_server_name)
        return {'message': f"MLflow Tracking Server '{tracking_server_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping MLflow Tracking Server {tracking_server_name}: {e}')
        raise ValueError(f'Failed to stop MLflow Tracking Server {tracking_server_name}: {e}')


@mcp.tool(name='delete_domain_sagemaker', description='Delete a SageMaker Domain')
async def delete_domain_sagemaker(
    domain_id: Annotated[str, Field(description='The ID of the SageMaker Domain to delete')],
) -> Dict[str, str]:
    """Delete a specified SageMaker Domain.

    ## Usage

    Use this tool to delete a SageMaker Domain by providing its ID. This is useful for cleaning
    up domains that are no longer needed.

    ## Example

    ```python
    result = await delete_domain_sagemaker(domain_id='d-1234567890')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_domain(domain_id)
        return {'message': f"Domain '{domain_id}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting domain {domain_id}: {e}')
        raise ValueError(f'Failed to delete domain {domain_id}: {e}')


@mcp.tool(name='list_domains_sagemaker', description='List all SageMaker Domains')
async def list_domains_sagemaker() -> Dict[str, List]:
    """List all SageMaker Domains.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Domains in your account in the current region.
    This is typically used to see what domains are available before performing operations on them.

    ## Example

    ```python
    domains = await list_domains_sagemaker()
    print(domains)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'domains': A list of dictionaries, each representing a SageMaker Domain with its details.

    ## Returns
    A dictionary containing a list of SageMaker Domains.
    """
    try:
        domains = await list_domains()
        return {'domains': domains}
    except Exception as e:
        logger.error(f'Error listing domains: {e}')
        raise ValueError(f'Failed to list domains: {e}')


@mcp.tool(name='describe_domain_sagemaker', description='Describe a SageMaker Domain')
async def describe_domain_sagemaker(
    domain_id: Annotated[str, Field(description='The ID of the SageMaker Domain to describe')],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Domain.

    ## Usage

    Use this tool to get detailed information about a SageMaker Domain by providing its ID.
    This returns comprehensive information about the domain's configuration, status, and other details.

    ## Example

    ```python
    domain_details = await describe_domain_sagemaker(domain_id='d-1234567890')
    print(domain_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker Domain.

    ## Returns
    A dictionary containing the domain details.
    """
    try:
        domain_details = await describe_domain(domain_id)
        return {'domain_details': domain_details}
    except Exception as e:
        logger.error(f'Error describing domain {domain_id}: {e}')
        raise ValueError(f'Failed to describe domain {domain_id}: {e}')


@mcp.tool(
    name='create_presigned_url_for_domain_sagemaker',
    description='Create a presigned URL for a SageMaker Domain',
)
async def create_presigned_url_for_domain_sagemaker(
    domain_id: Annotated[str, Field(description='The ID of the SageMaker Domain')],
    user_profile_name: Annotated[str, Field(description='The name of the user profile')],
    expiration_seconds: Annotated[
        int, Field(description='The expiration time for the presigned URL in seconds')
    ] = 3600,
) -> Dict[str, str]:
    """Create a presigned URL for accessing a SageMaker Domain.

    ## Usage

    Use this tool to create a presigned URL for a SageMaker Domain by providing its ID and user profile name.
    This is useful for granting temporary access to the domain.

    ## Example

    ```python
    presigned_url = await create_presigned_url_for_domain_sagemaker(
        domain_id='d-1234567890', user_profile_name='test-user', expiration_seconds=3600
    )
    print(presigned_url)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'presigned_url': The presigned URL for the SageMaker Domain.

    ## Returns
    A dictionary containing the presigned URL.
    """
    try:
        presigned_url = await create_presigned_domain_url(
            domain_id,
            user_profile_name,
            expiration_seconds,
        )
        return {'presigned_url': presigned_url}
    except Exception as e:
        logger.error(f'Error creating presigned URL for domain {domain_id}: {e}')
        raise ValueError(f'Failed to create presigned URL for domain {domain_id}: {e}')


@mcp.tool(name='list_spaces_sagemaker', description='List all SageMaker Spaces')
async def list_spaces_sagemaker() -> Dict[str, List]:
    """List all SageMaker Spaces.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Spaces in your account in the current region.
    This is typically used to see what spaces are available before performing operations on them.

    ## Example

    ```python
    spaces = await list_spaces_sagemaker()
    print(spaces)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'spaces': A list of dictionaries, each representing a SageMaker Space with its details.

    ## Returns
    A dictionary containing a list of SageMaker Spaces.
    """
    try:
        spaces = await list_spaces()
        return {'spaces': spaces}
    except Exception as e:
        logger.error(f'Error listing spaces: {e}')
        raise ValueError(f'Failed to list spaces: {e}')


@mcp.tool(name='list_user_profiles_sagemaker', description='List all SageMaker User Profiles')
async def list_user_profiles_sagemaker() -> Dict[str, List]:
    """List all SageMaker User Profiles.

    ## Usage

    Use this tool to retrieve a list of all SageMaker User Profiles in your account in the current region.
    This is typically used to see what user profiles are available before performing operations on them.

    ## Example

    ```python
    user_profiles = await list_user_profiles_sagemaker()
    print(user_profiles)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'user_profiles': A list of dictionaries, each representing a SageMaker User Profile with its details.

    ## Returns
    A dictionary containing a list of SageMaker User Profiles.
    """
    try:
        user_profiles = await list_user_profiles()
        return {'user_profiles': user_profiles}
    except Exception as e:
        logger.error(f'Error listing user profiles: {e}')
        raise ValueError(f'Failed to list user profiles: {e}')


@mcp.tool(name='describe_model_sagemaker', description='Describe a SageMaker Model')
async def describe_model_sagemaker(
    model_name: Annotated[str, Field(description='The name of the SageMaker Model to describe')],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Model.

    ## Usage

    Use this tool to get detailed information about a SageMaker Model by providing its name.
    This returns comprehensive information about the model's configuration, status, and other details.

    ## Example

    ```python
    model_details = await describe_model_sagemaker(model_name='my-model')
    print(model_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker Model.

    ## Returns
    A dictionary containing the model details.
    """
    try:
        model_details = await describe_model(model_name)
        return {'model_details': model_details}
    except Exception as e:
        logger.error(f'Error describing model {model_name}: {e}')
        raise ValueError(f'Failed to describe model {model_name}: {e}')


@mcp.tool(name='list_models_sagemaker', description='List all SageMaker Models')
async def list_models_sagemaker() -> Dict[str, List]:
    """List all SageMaker Models.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Models in your account in the current region.
    This is typically used to see what models are available before performing operations on them.

    ## Example

    ```python
    models = await list_models_sagemaker()
    print(models)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'models': A list of dictionaries, each representing a SageMaker Model with its details.

    ## Returns
    A dictionary containing a list of SageMaker Models.
    """
    try:
        models = await list_models()
        return {'models': models}
    except Exception as e:
        logger.error(f'Error listing models: {e}')
        raise ValueError(f'Failed to list models: {e}')


@mcp.tool(name='delete_model_sagemaker', description='Delete a SageMaker Model')
async def delete_model_sagemaker(
    model_name: Annotated[str, Field(description='The name of the SageMaker Model to delete')],
) -> Dict[str, str]:
    """Delete a specified SageMaker Model.

    ## Usage

    Use this tool to delete a SageMaker Model by providing its name. This is useful for cleaning up
    models that are no longer needed.

    ## Example

    ```python
    result = await delete_model_sagemaker(model_name='my-model')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_model(model_name)
        return {'message': f"Model '{model_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting model {model_name}: {e}')
        raise ValueError(f'Failed to delete model {model_name}: {e}')


@mcp.tool(name='describe_model_card_sagemaker', description='Describe a SageMaker Model Card')
async def describe_model_card_sagemaker(
    model_card_name: Annotated[
        str, Field(description='The name of the SageMaker Model Card to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Model Card.

    ## Usage

    Use this tool to get detailed information about a SageMaker Model Card by providing its name.
    This returns comprehensive information about the model card's configuration, status, and other details.

    ## Example

    ```python
    model_card_details = await describe_model_card_sagemaker(model_card_name='my-model-card')
    print(model_card_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker Model Card.

    ## Returns
    A dictionary containing the model card details.
    """
    try:
        model_card_details = await describe_model_card(model_card_name)
        return {'model_card_details': model_card_details}
    except Exception as e:
        logger.error(f'Error describing model card {model_card_name}: {e}')
        raise ValueError(f'Failed to describe model card {model_card_name}: {e}')


@mcp.tool(name='delete_model_card_sagemaker', description='Delete a SageMaker Model Card')
async def delete_model_card_sagemaker(
    model_card_name: Annotated[
        str, Field(description='The name of the SageMaker Model Card to delete')
    ],
) -> Dict[str, str]:
    """Delete a specified SageMaker Model Card.

    ## Usage

    Use this tool to delete a SageMaker Model Card by providing its name. This is useful for cleaning up
    model cards that are no longer needed.

    ## Example

    ```python
    result = await delete_model_card_sagemaker(model_card_name='my-model-card')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await delete_model_card(model_card_name)
        return {'message': f"Model Card '{model_card_name}' deleted successfully"}
    except Exception as e:
        logger.error(f'Error deleting model card {model_card_name}: {e}')
        raise ValueError(f'Failed to delete model card {model_card_name}: {e}')


@mcp.tool(name='list_model_cards_sagemaker', description='List all SageMaker Model Cards')
async def list_model_cards_sagemaker() -> Dict[str, List]:
    """List all SageMaker Model Cards.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Model Cards in your account in the current region.
    This is typically used to see what model cards are available before performing operations on them.

    ## Example

    ```python
    model_cards = await list_model_cards_sagemaker()
    print(model_cards)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'model_cards': A list of dictionaries, each representing a SageMaker Model Card with its details.

    ## Returns
    A dictionary containing a list of SageMaker Model Cards.
    """
    try:
        model_cards = await list_model_cards()
        return {'model_cards': model_cards}
    except Exception as e:
        logger.error(f'Error listing model cards: {e}')
        raise ValueError(f'Failed to list model cards: {e}')


@mcp.tool(
    name='list_model_card_export_jobs_sagemaker',
    description='List Model Card Export Jobs for a SageMaker Model Card',
)
async def list_model_card_export_jobs_sagemaker(
    model_card_name: Annotated[
        str, Field(description='The name of the SageMaker Model Card to list export jobs for')
    ],
) -> Dict[str, List]:
    """List Model Card Export Jobs for a specified SageMaker Model Card.

    ## Usage

    Use this tool to retrieve a list of all export jobs for a specific SageMaker Model Card by providing its name.
    This helps you track the export jobs associated with the model card.

    ## Example

    ```python
    export_jobs = await list_model_card_export_jobs_sagemaker(model_card_name='my-model-card')
    print(export_jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'model_card_export_jobs': A list of dictionaries, each representing an export job for the SageMaker Model Card.

    ## Returns
    A dictionary containing a list of Model Card Export Jobs.
    """
    try:
        export_jobs = await list_model_card_export_jobs(model_card_name)
        return {'model_card_export_jobs': export_jobs}
    except Exception as e:
        logger.error(f'Error listing model card export jobs for {model_card_name}: {e}')
        raise ValueError(f'Failed to list model card export jobs for {model_card_name}: {e}')


@mcp.tool(
    name='list_model_card_versions_sagemaker',
    description='List all versions of a SageMaker Model Card',
)
async def list_model_card_versions_sagemaker(
    model_card_name: Annotated[
        str, Field(description='The name of the SageMaker Model Card to list versions for')
    ],
) -> Dict[str, List]:
    """List all versions of a SageMaker Model Card.

    ## Usage

    Use this tool to retrieve a list of all versions for a specific SageMaker Model Card by providing its name.
    This helps you track the different versions of the model card.

    ## Example

    ```python
    model_card_versions = await list_model_card_versions_sagemaker(model_card_name='my-model-card')
    print(model_card_versions)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'model_card_versions': A list of dictionaries, each representing a version of the SageMaker Model Card.

    ## Returns
    A dictionary containing a list of Model Card Versions.
    """
    try:
        model_card_versions = await list_model_card_versions(model_card_name)
        return {'model_card_versions': model_card_versions}
    except Exception as e:
        logger.error(f'Error listing model card versions for {model_card_name}: {e}')
        raise ValueError(f'Failed to list model card versions for {model_card_name}: {e}')


@mcp.tool(
    name='list_inference_recommendations_jobs_sagemaker',
    description='List all SageMaker Inference Recommender Jobs',
)
async def list_inference_recommendations_jobs_sagemaker() -> Dict[str, List]:
    """List all SageMaker Inference Recommender Jobs.

    ## Usage

    Use this tool to retrieve a list of all SageMaker Inference Recommender Jobs
    in your account in the current region. This is typically used to see what
    inference recommender jobs are available before performing operations on them.

    ## Example

    ```python
    jobs = await list_inference_recommendations_jobs_sagemaker()
    print(jobs)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'inference_recommendations_jobs': A list of dictionaries, each representing
      a SageMaker Inference Recommender Job with its details.

    ## Returns
    A dictionary containing a list of SageMaker Inference Recommender Jobs.
    """
    try:
        jobs = await list_inference_recommendations_jobs()
        return {'inference_recommendations_jobs': jobs}
    except Exception as e:
        logger.error(f'Error listing inference recommender jobs: {e}')
        raise ValueError(f'Failed to list inference recommender jobs: {e}')


@mcp.tool(
    name='list_inference_recommendations_job_steps_sagemaker',
    description='List steps for a SageMaker Inference Recommender Job',
)
async def list_inference_recommendations_job_steps_sagemaker(
    job_name: Annotated[
        str,
        Field(description='The name of the SageMaker Inference Recommender Job to list steps for'),
    ],
) -> Dict[str, List]:
    """List steps for a specific SageMaker Inference Recommender Job.

    ## Usage

    Use this tool to retrieve a list of steps for a specific SageMaker Inference
    Recommender Job by providing its name. This helps you track the progress of the job.

    ## Example

    ```python
    steps = await list_inference_recommendations_job_steps_sagemaker(job_name='my-inference-job')
    print(steps)
    ```

    ## Output Format

    The output is a dictionary with the following structure:
    - 'steps': A list of dictionaries, each representing a step in the SageMaker
      Inference Recommender Job with its details.

    ## Returns
    A dictionary containing a list of steps for the specified Inference Recommender Job.
    """
    try:
        steps = await list_inference_recommendations_job_steps(job_name)
        return {'steps': steps}
    except Exception as e:
        logger.error(f'Error listing steps for inference recommender job {job_name}: {e}')
        raise ValueError(f'Failed to list steps for inference recommender job {job_name}: {e}')


@mcp.tool(
    name='describe_inference_recommendations_job_sagemaker',
    description='Describe a SageMaker Inference Recommender Job',
)
async def describe_inference_recommendations_job_sagemaker(
    job_name: Annotated[
        str, Field(description='The name of the SageMaker Inference Recommender Job to describe')
    ],
) -> Dict[str, Any]:
    """Describe a specified SageMaker Inference Recommender Job.

    ## Usage

    Use this tool to get detailed information about a SageMaker Inference
    Recommender Job by providing its name. This returns comprehensive information
    about the job's configuration, status, and other details.

    ## Example

    ```python
    job_details = await describe_inference_recommendations_job_sagemaker(
        job_name='my-inference-job'
    )
    print(job_details)
    ```

    ## Output Format

    The output is a dictionary containing all the details of the SageMaker
    Inference Recommender Job.

    ## Returns
    A dictionary containing the job details.
    """
    try:
        job_details = await describe_inference_recommendations_job(job_name)
        return {'job_details': job_details}
    except Exception as e:
        logger.error(f'Error describing inference recommender job {job_name}: {e}')
        raise ValueError(f'Failed to describe inference recommender job {job_name}: {e}')


@mcp.tool(
    name='stop_inference_recommendations_job_sagemaker',
    description='Stop a SageMaker Inference Recommender Job',
)
async def stop_inference_recommendations_job_sagemaker(
    job_name: Annotated[
        str, Field(description='The name of the SageMaker Inference Recommender Job to stop')
    ],
) -> Dict[str, str]:
    """Stop a specified SageMaker Inference Recommender Job.

    ## Usage

    Use this tool to stop a SageMaker Inference Recommender Job by providing its name.
    This is useful for stopping jobs that are no longer needed or are consuming
    too many resources.

    ## Example

    ```python
    result = await stop_inference_recommendations_job_sagemaker(job_name='my-inference-job')
    print(result)
    ```

    ## Output Format

    The output is a dictionary with a success message.

    ## Returns
    A dictionary containing a success message.
    """
    try:
        await stop_inference_recommendations_job(job_name)
        return {'message': f"Inference Recommender Job '{job_name}' stopped successfully"}
    except Exception as e:
        logger.error(f'Error stopping inference recommender job {job_name}: {e}')
        raise ValueError(f'Failed to stop inference recommender job {job_name}: {e}')


def main():
    """Run the SageMaker AI MCP Server."""
    logger.info('Welcome to the SageMaker AI MCP Server!')
    mcp.run()


if __name__ == '__main__':
    main()
