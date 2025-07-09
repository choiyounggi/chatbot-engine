package com.yk.chatbot.lasa;

import java.util.Map;

/**
 * Analyze 인터페이스는 LASA 프레임워크의 두 번째 단계로,
 * 사용자 입력을 분석하여 의도와 엔티티를 추출하는 역할을 담당합니다.
 */
public interface Analyze {
    /**
     * 사용자 메시지를 분석하여 의도와 엔티티를 추출합니다.
     * @param message 사용자 메시지
     * @return 의도와 추출된 엔티티를 포함한 분석 결과
     */
    AnalysisResult analyze(String message);
}
