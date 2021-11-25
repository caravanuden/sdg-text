#!/bin/sh#
# Before running this script, you need to:
# 1) download gcloud cli
# 2) call gcloud auth login <your_google_account_name>
# 3) create a google project with id <project_id>
# 4) gcloud config set project <project_id>


read -p "Please enter a service account name: " SA_ACCT_NAME
# read -p "Please enter the path to the JSON credentials for the service account" PATH_TO_CREDENTIALS
read -p "Please enter the Google Cloud project ID: " PROJECT_ID


# export GOOGLE_APPLICATION_CREDENTIALS=$PATH_TO_CREDENTIALS
gcloud iam service-accounts create $SA_ACCT_NAME 
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:$SA_ACCT_NAME@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/owner"

if [[ ! -e $dir ]]; then
  mkdir service_account_keys
fi

gcloud iam service-accounts keys create service_account_keys/${SA_ACCT_NAME}_key.json --iam-account=${SA_ACCT_NAME}@$PROJECT_ID.iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS=`$pwd`service_account_keys/${SA_ACCT_NAME}_key.json
