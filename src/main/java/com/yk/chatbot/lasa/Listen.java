package com.yk.chatbot.lasa;

import com.yk.chatbot.dto.ChatRequest;

/**
 * Listen 인터페이스 - 사용자 입력을 처리하는 첫 번째 단계
 */
public interface Listen {
    
    /**
     * 사용자 입력을 수신하고 전처리합니다.
     * 
     * @param request 사용자 요청 정보
     * @return 전처리된 메시지 문자열
     */
    String listen(ChatRequest request);
}
