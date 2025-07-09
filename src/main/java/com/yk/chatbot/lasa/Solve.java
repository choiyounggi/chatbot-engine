package com.yk.chatbot.lasa;

/**
 * Solve 인터페이스는 LASA 프레임워크의 세 번째 단계로,
 * 분석된 사용자 의도에 따라 문제를 해결하는 역할을 담당합니다.
 */
public interface Solve {
    /**
     * 분석 결과에 기반하여 사용자 요청을 처리하고 응답을 준비합니다.
     * @param result 분석 결과
     * @return 응답 정보를 포함한 해결 결과
     */
    SolutionResult solve(AnalysisResult result);
}
