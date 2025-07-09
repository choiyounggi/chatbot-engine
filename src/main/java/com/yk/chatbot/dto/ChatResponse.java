package com.yk.chatbot.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * 채팅 응답 DTO
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponse {
    
    /**
     * 챗봇 응답 메시지
     */
    private String message;
    
    /**
     * 인식된 의도
     */
    private String intent;
    
    /**
     * 추출된 엔티티들
     */
    private Map<String, Object> entities;
    
    /**
     * 응답 생성 시간 (timestamp)
     */
    private long timestamp;
    
    /**
     * 신뢰도 점수 (Rasa NLU에서 제공)
     */
    private double confidence;
}
