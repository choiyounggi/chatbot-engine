# LASA-based Chatbot Engine

**English** | [한국어](README.ko.md)

A chatbot engine built on the LASA (Listen-Analyze-Solve-Answer) framework.

## What is LASA?

LASA is a framework for building effective chatbot systems, composed of four stages:

1. **Listen**: receive and preprocess the user's input.
2. **Analyze**: extract the user's intent and related entities.
3. **Solve**: resolve the analyzed intent and shape a response strategy.
4. **Answer**: generate the final response and deliver it to the user.

## System structure

```
com.yk.chatbot
├── controller
│   └── ChatbotController.java    # REST API controller
├── dto
│   ├── ChatRequest.java          # request DTO
│   └── ChatResponse.java         # response DTO
├── lasa                          # LASA framework interfaces
│   ├── Listen.java               # listen interface
│   ├── Analyze.java              # analyze interface
│   ├── AnalysisResult.java       # analysis result class
│   ├── Solve.java                # solve interface
│   ├── SolutionResult.java       # solution result class
│   ├── Answer.java               # answer interface
│   └── impl                      # implementations
│       ├── SimpleListener.java   # default listener
│       ├── SimpleAnalyzer.java   # default analyzer
│       ├── SimpleSolver.java     # default solver
│       └── SimpleAnswerer.java   # default answerer
└── service
    └── LasaChatbotService.java   # LASA-based chatbot service
```

## API usage

### Request

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "How's the weather today?"
}
```

### Response

```json
{
  "reply": "It is currently sunny in Seoul, 23°C.",
  "intent": "weather"
}
```

## Extending

To add a new intent or extend functionality:

1. Add the new intent pattern in `SimpleAnalyzer`
2. Implement handling logic for the new intent in `SimpleSolver`
3. Add new DTOs / entity types as needed

## Dependencies

- Spring Boot
- Lombok
