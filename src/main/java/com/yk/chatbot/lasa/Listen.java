package com.yk.chatbot.lasa;

import com.yk.chatbot.dto.ChatRequest;

/**
 * Listen 인터페이스는 LASA 프레임워크의 첫 단계로,
 * 사용자 입력을 받고 전처리하는 역할을 담당합니다.
 */
public interface Listen {
    /**
     * 사용자 입력을 받아 전처리한 후 처리 가능한 형태로 변환합니다.
     * @param request 사용자 요청 정보
     * @return 전처리된 사용자 메시지
     */
    String listen(ChatRequest request);
}
