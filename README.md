NxP-Agentic-Geoguesser
NXP-CTRL Project 2

Autonomous AI agents powered by AWS Bedrock, Rekognition, Lambda, API Gateway, and S3.

📌 Overview

This project implements agentic AI workflows using AWS-managed services.

Agents can:

🧠 Generate text and reasoning using Amazon Bedrock (LLMs)

🖼 Analyze and label images using Amazon Rekognition

⚡ Execute backend logic with AWS Lambda

🌍 Expose Wikipedia Geosearch APIs via API Gateway

🌐 Serve a web interface using Amazon S3 static hosting

The architecture is serverless, scalable, and production-ready.

🏗 Architecture
User (Browser / Client)
        │
        ▼
Amazon S3 (Static Frontend Hosting)
        │
        ▼
API Gateway (HTTP Endpoint)
        │
        ▼
AWS Lambda (Agent Logic)
        │
        ├── Amazon Bedrock (LLM reasoning + generation)
        └── Amazon Rekognition (Image analysis)

🧩 Service Responsibilities
Service	Purpose
Amazon Bedrock	LLM reasoning, text generation, planning
Amazon Rekognition	Image detection and labeling
AWS Lambda	Core agent logic and orchestration
API Gateway	Public HTTP interface
Amazon S3	Static frontend hosting        

🚀 Features

🔁 Agent reasoning loops

🧠 LLM-powered decision making

🖼 Image understanding

⚡ Serverless backend

🌐 HTTP API interface

📦 Static frontend hosting

⚙️ Local Development Setup

1️⃣ Install uv (Fast Python Package Manager)

uv replaces pip and venv management.

Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

Verify installation:

uv --version
2️⃣ Create Project Environment

Inside your project root:

uv venv
3️⃣ Activate Virtual Environment
Mac / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate