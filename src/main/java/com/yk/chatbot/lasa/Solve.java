package com.yk.chatbot.lasa;

/**
 * Solve 인터페이스 - 분석 결과를 바탕으로 문제를 해결하는 세 번째 단계
 * 분석된 의도와 엔티티를 활용하여 적절한 비즈니스 로직 수행
 */
public interface Solve {
    
    /**
     * 분석 결과를 바탕으로 문제를 해결하고 응답 전략을 수립합니다.
     * 
     * @param result 분석 결과
     * @return 해결 결과 (응답 템플릿, 데이터 등)
     */
    SolutionResult solve(AnalysisResult result);
}
