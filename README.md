# BMW X Series Manual Chatbot

<div>
  <img src="https://cdn.smartrep.gr/uni-ai/smartrep-logo.png" alt="SmartRep" height="100"/>
  <img src="https://cdn.smartrep.gr/uni-ai/uniai-logo.png" alt="Uni AI" height="100"/>
</div>

A chatbot application that can answer questions about the BMW X1 manual using LLM technology.

## Hackathon Challenge

1. Create a document-based Q&A system using LLM technology
2. Process and understand a technical manual (BMW X1)

## Project Structure

- `frontend/`: React application with a chat interface
- `backend/`: Python FastAPI server for LLM integration
- `data/`: Directory to store the BMW X Series manual PDF
- `docker-compose.yml`: Docker Compose configuration for the entire application

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- BMW X1 Series manual PDF

### Configuration

1. Create a `.env` file in the root directory with your API keys e.g. :
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Application

1. Build and start the containers:
   ```
   docker-compose up -d --build
   ```

2. Access the chatbot at http://localhost

3. To view logs
   ```
   docker compose logs -f backend
   ```

4. To stop the application:
   ```
   docker-compose down
   ```
