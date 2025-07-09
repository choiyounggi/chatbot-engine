package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.lasa.Listen;
import org.springframework.stereotype.Component;

/**
 * 간단한 리스너 구현체
 * 사용자 입력을 전처리합니다.
 */
@Component
public class SimpleListener implements Listen {

    @Override
    public String listen(ChatRequest request) {
        if (request == null || request.getMessage() == null) {
            return "";
        }
        
        // 앞뒤 공백 제거 및 연속된 공백 처리
        String message = request.getMessage().trim().replaceAll("\\s+", " ");
        
        // 특수 문자 정리 (선택적)
        message = message.replaceAll("[^\\p{L}\\p{N}\\s\\p{Punct}]", "");
        
        return message;
    }
}
