#!/bin/bash
# Navigate to the deployment directory
cd /home/ec2-user/ml-model

# Install dependencies
sudo pip install -r requirements.txt

# Kill any existing Streamlit process (optional)
sudo pkill -f streamlit || true

# Run the Streamlit application
nohup streamlit run app.py --server.headless true > streamlit.log 2>&1 &
