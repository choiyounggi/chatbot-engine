package com.yk.chatbot.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 채팅 요청 DTO
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatRequest {
    
    /**
     * 사용자가 입력한 메시지
     */
    private String message;
    
    /**
     * 사용자 식별자 (선택적)
     */
    private String userId;
    
    /**
     * 세션 식별자 (선택적)
     */
    private String sessionId;
}
