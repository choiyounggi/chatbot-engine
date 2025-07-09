package com.yk.chatbot.lasa;

/**
 * Analyze 인터페이스 - 메시지 분석을 담당하는 두 번째 단계
 * 한국어 의도 분석 및 엔티티 추출을 수행합니다.
 */
public interface Analyze {
    
    /**
     * 메시지를 분석하여 의도와 엔티티를 추출합니다.
     * 
     * @param message 사용자 메시지
     * @return 분석 결과 (의도, 엔티티, 신뢰도 등)
     */
    AnalysisResult analyze(String message);
    
    /**
     * 사용자 ID를 포함한 메시지 분석
     * 
     * @param message 사용자 메시지
     * @param userId 사용자 ID
     * @return 분석 결과
     */
    default AnalysisResult analyze(String message, String userId) {
        return analyze(message);
    }
}
