package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.dto.ChatRequest;
import com.yk.chatbot.lasa.Listen;
import org.springframework.stereotype.Component;

/**
 * Listen 인터페이스의 기본 구현체입니다.
 * 사용자 입력을 정규화하고 기본적인 전처리를 수행합니다.
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
