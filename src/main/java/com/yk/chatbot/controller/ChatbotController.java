package com.yk.chatbot.controller;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.service.LasaChatbotService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/chat")
@RequiredArgsConstructor
public class ChatbotController {

    private final LasaChatbotService lasaChatbotService;

    @PostMapping
    public ChatResponse chat(@RequestBody ChatRequest request) {
        return lasaChatbotService.process(request);
    }
}