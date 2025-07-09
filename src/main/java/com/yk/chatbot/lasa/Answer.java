package com.yk.chatbot.lasa;

import com.yk.chatbot.dto.ChatResponse;

/**
 * Answer 인터페이스는 LASA 프레임워크의 마지막 단계로,
 * 해결 결과를 사용하여 최종 응답을 생성하는 역할을 담당합니다.
 */
public interface Answer {
    /**
     * 해결 결과를 바탕으로 사용자에게 보여줄 최종 응답을 생성합니다.
     * @param result 해결 결과
     * @return 사용자에게 전달될 최종 응답
     */
    ChatResponse answer(SolutionResult result);
}
