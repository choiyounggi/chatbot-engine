package com.yk.chatbot.service;

import io.github.sashirestela.cleverclient.client.OkHttpClientAdapter;
import io.github.sashirestela.openai.SimpleOpenAI;
import io.github.sashirestela.openai.domain.chat.ChatMessage;
import io.github.sashirestela.openai.domain.chat.ChatRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

/**
 * OpenAI API를 통해 챗봇 응답을 개선하는 서비스
 */
@Slf4j
@Service
@EnableAsync
public class OpenAIService {

    @Value("${openai.key}")
    private String openAIApiKey;
    
    @Value("${openai.model}")
    private String model;
    
    @Value("${openai.enabled}")
    private boolean enabled;

    @Value("${openai.max-tokens}")
    private int maxTokens;

    @Value("${openai.temperature}")
    private double temperature;

    /**
     * 사용자 메시지에 대한 응답을 비동기적으로 생성합니다.
     * 
     * @param userMessage 사용자 메시지
     * @return 생성된 응답이 포함된 CompletableFuture 또는 API가 비활성화된 경우 기본 메시지
     */
    @Async
    public CompletableFuture<String> generateResponseAsync(String userMessage) {
        if (!enabled || openAIApiKey == null || openAIApiKey.isEmpty()) {
            log.warn("OpenAI 서비스가 비활성화되었거나 API 키가 없습니다.");
            return CompletableFuture.completedFuture("죄송합니다. 요청을 이해하지 못했습니다. 다른 방식으로 질문해 주시겠어요?");
        }
        
        try {
            log.info("OpenAI API 호출: 사용자 메시지={}", userMessage);
            
            SimpleOpenAI openAI = SimpleOpenAI.builder()
                    .apiKey(openAIApiKey)
                    .clientAdapter(new OkHttpClientAdapter())
                    .build();

            ChatRequest req = ChatRequest.builder()
                    .model(model)
                    .message(ChatMessage.SystemMessage.of("너는 사용자의 궁금증을 해결해주는 챗봇이야. 사용자의 질문에 대해 정확하고 간결하게 답변해줘."))
                    .message(ChatMessage.UserMessage.of(userMessage))
                    .temperature(temperature)
                    .maxCompletionTokens(maxTokens)
                    .build();

            String response = openAI.chatCompletions().create(req).join().firstContent();
            
            log.info("OpenAI 응답 생성 완료");
            return CompletableFuture.completedFuture(response);
            
        } catch (Exception e) {
            log.error("OpenAI API 호출 중 오류 발생", e);
            return CompletableFuture.completedFuture("죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요.");
        }
    }
}