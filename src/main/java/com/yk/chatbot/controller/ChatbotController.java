package com.yk.chatbot.controller;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.dto.ChatResponse;
import com.yk.chatbot.service.LasaChatbotService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.validation.Valid;

/**
 * 챗봇 API 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@Tag(name = "챗봇 API", description = "챗봇 메시지 처리 API")
public class ChatbotController {

    private final LasaChatbotService chatbotService;

    /**
     * 챗봇 메시지 처리 API
     * 
     * @param request 사용자 요청
     * @return 챗봇 응답
     */
    @PostMapping
    @Operation(
        summary = "메시지 처리",
        description = "사용자 메시지를 처리하고 챗봇 응답을 반환합니다."
    )
    public ResponseEntity<ChatResponse> processMessage(@Valid @RequestBody ChatRequest request) {
        log.info("메시지 요청 수신: {}", request);
        
        ChatResponse response = chatbotService.process(request);
        
        return ResponseEntity.ok(response);
    }
}
