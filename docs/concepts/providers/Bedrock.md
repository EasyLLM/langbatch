# Bedrock

You can run batch inference jobs on Claude and Nova models available in Bedrock via LangBatch.

## Data Format

OpenAI data format can be used in LangBatch for Bedrock. But the model name can be skipped here.

```json
{"custom_id": "task-0", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was Microsoft founded?"}]}}
{"custom_id": "task-1", "method": "POST", "url": "/chat/completions", "body": {"messages": [{"role": "system", "content": "You are an AI assistant that helps people find information."}, {"role": "user", "content": "When was the first XBOX released?"}]}}
```

???+note
    In Bedrock, you can only send requests to a single model in a batch. If you want to use multiple models, you need to create multiple batches.

## Bedrock Setup

???+ info
    Make sure you have the access to the Foundation Models, follow this guide to get access to the Foundation Models: [Getting Access to Bedrock Foundation Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html){:target="_blank"}

To use Bedrock, you need to setup few things. Please follow these steps:

- You need to set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables (From user with [Bedrock Batch Inference permissions](#user-permissions))
- Create two new S3 buckets. One for storing batch input and another one for storing batch output: For example `batch-input` and `batch-output`
- And you need to create a [service role](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html){:target="_blank"} using the instructions [here](#create-service-role)

???+note
    You need to use the correct region according to the model you are using. Check this [link](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html){:target="_blank"} for more available regions. Also, you need to create new S3 buckets for each region.

## Create Chat Completion Batch

```python
import os
from langbatch import chat_completion_batch

# Set the access credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key-id"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-access-key"

# Set the configuration values
os.environ["AWS_INPUT_BUCKET"] = "your-input-bucket"
os.environ["AWS_OUTPUT_BUCKET"] = "your-output-bucket"
os.environ["AWS_REGION"] = "your-aws-region" # us-west-2
os.environ["AWS_SERVICE_ROLE"] = "your-service-role-arn"

batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="bedrock",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)
```

You can also pass the configuration values as arguments:

```python
batch = chat_completion_batch(
    "path/to/batch-file.jsonl", 
    provider="bedrock",
    model="us.amazon.nova-pro-v1:0",
    input_bucket="your-input-bucket",
    output_bucket="your-output-bucket",
    region="your-aws-region", # us-east-1
    service_role="your-service-role-arn"
)
```

### Supported Models
You need to enable the models you want to use in Bedrock before using them. 

- Claude 3.5 Sonnet v2, Claude 3.5 Haiku
- Nova Lite, Nova Micro, Nova Pro

## Create Service Role

1. Go to Identity and Access Management (IAM)
2. Click on "Policies" on the left sidebar -> Click on "Create policy" -> Click on "JSON" tab -> Add the following JSON. Modify the bucket names accordingly -> Click on "Next" -> Provide the Name (For example `RolePolicyForBedrockBatchInference`) -> Click on "Create Policy"
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{batch-input-bucket-name}",
                "arn:aws:s3:::{batch-input-bucket-name}/*",
                "arn:aws:s3:::{batch-output-bucket-name}",
                "arn:aws:s3:::{batch-output-bucket-name}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "*"
        }
    ]
}
```
3. Click on "Roles" on the left sidebar -> Click on "Create role" -> Click on "Custom trust policy" -> Paste the below JSON into Custom trust policy field-> Click on "Next" -> Click on "Next"
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```
4. In the Add permissions section, Search for the policy you just created in the previous step and Select it. Then, Click on "Next".
5. Provide the Role Name (For example `BedrockBatchInferenceServiceRole`) and Click on "Create role"
6. Copy the Role ARN and use it in the Bedrock Batch creation.

???+tip
    If you are using multiple regions, include the region specific S3 buckets in the policy.

## User Permissions
???+note
    Make sure the AWS User you are using has access to Bedrock Batch Inference permissions. You can create the below policy and attach it to the user.

1. Go to Identity and Access Management (IAM)
2. Click on "Policies" on the left sidebar -> Click on "Create policy" -> Click on "JSON" tab -> Add the following JSON. Modify the bucket names accordingly -> Click on "Next" -> Provide the Name (For example `UserPolicyForBedrockBatchInference`) -> Click on "Create Policy"
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:GetFoundationModel",
                "bedrock:GetFoundationModelAvailability",
                "bedrock:ListFoundationModels",
                "bedrock:InvokeModel",
                "bedrock:StopModelInvocationJob",
                "bedrock:CreateModelInvocationJob",
                "bedrock:ListModelInvocationJobs",
                "bedrock:GetModelInvocationJob"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::{batch-input-bucket-name}/*",
                "arn:aws:s3:::{batch-input-bucket-name}",
                "arn:aws:s3:::{batch-output-bucket-name}/*",
                "arn:aws:s3:::{batch-output-bucket-name}"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": [
                "arn:aws:iam::{aws_account_id}:role/BedrockBatchInferenceServiceRole"
            ]
        }
    ]
}
```
3. Click on "Users" on the left sidebar -> Click on the user you want to attach the policy to -> Click on "Add permissions" -> Click on "Attach existing policies directly" -> Click on "Next" -> Click on "Add permissions"