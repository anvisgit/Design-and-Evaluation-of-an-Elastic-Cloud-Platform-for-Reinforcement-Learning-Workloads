#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-dev}"
OWNER_EMAIL="${2:-${OWNER_EMAIL:-}}"
CTX=(--context "stage=${STAGE}" --context "owner_email=${OWNER_EMAIL}")

command -v cdk >/dev/null    || { echo "install the CDK CLI: npm install -g aws-cdk"; exit 1; }
command -v docker >/dev/null || { echo "docker is required to build the worker image"; exit 1; }
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region)}}
echo ">> account=$ACCOUNT region=$REGION stage=$STAGE"

echo ">> unit tests";           make test
echo ">> Lambda dependency layer"; make layer
echo ">> bootstrap (idempotent)"; cdk bootstrap "aws://$ACCOUNT/$REGION" || true
echo ">> Storage + Messaging";  cdk deploy StorageStack MessagingStack "${CTX[@]}" --require-approval never
echo ">> worker image";         make push-image
echo ">> Api + Compute + Monitoring"
cdk deploy ApiStack ComputeStack MonitoringStack "${CTX[@]}" --require-approval never \
    --outputs-file build/outputs.json

API=$(python -c "import json;print(json.load(open('build/outputs.json'))['ApiStack']['ApiUrl'])")
POOL=$(python -c "import json;print(json.load(open('build/outputs.json'))['ApiStack']['UserPoolId'])")
CLIENT=$(python -c "import json;print(json.load(open('build/outputs.json'))['ApiStack']['UserPoolClientId'])")
cat <<MSG

Deployed.
  API url      : $API
  User pool    : $POOL     client id: $CLIENT
  Dashboard    : see MonitoringStack output DashboardUrl

Next steps
  1. Confirm the SNS subscription email (check ${OWNER_EMAIL:-your inbox}).
  2. Create a user (sign-up is disabled):
       aws cognito-idp admin-create-user --user-pool-id $POOL --username me --message-action SUPPRESS
       aws cognito-idp admin-set-user-password --user-pool-id $POOL --username me --password 'Change-me-12345' --permanent
  3. python scripts/smoke_test.py --api $API --client-id $CLIENT --region $REGION --username me --password 'Change-me-12345' --runs 3
MSG
