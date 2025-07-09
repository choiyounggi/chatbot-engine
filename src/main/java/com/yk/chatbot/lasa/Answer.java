package com.yk.chatbot.lasa;

import com.yk.chatbot.dto.ChatResponse;

/**
 * Answer 인터페이스 - 해결 결과를 바탕으로 최종 응답을 생성하는 네 번째 단계
 */
public interface Answer {
    
    /**
     * 해결 결과를 바탕으로 최종 응답을 생성합니다.
     * 
     * @param result 해결 결과
     * @return 최종 응답
     */
    ChatResponse answer(SolutionResult result);
}
