package com.yk.chatbot.lasa;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.Map;

/**
 * 분석 결과 클래스
 * Rasa NLU 등을 통해 분석된 결과를 저장합니다.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AnalysisResult {
    
    /**
     * 분석된 사용자 의도
     */
    private String intent;
    
    /**
     * 추출된 엔티티 (key: 엔티티명, value: 엔티티 값)
     */
    @Builder.Default
    private Map<String, String> entities = new HashMap<>();
    
    /**
     * 의도 분석 신뢰도 점수 (0.0 ~ 1.0)
     */
    private double confidence;
    
    /**
     * 원본 메시지
     */
    private String originalMessage;
    
    /**
     * 엔티티 추가 헬퍼 메소드
     */
    public AnalysisResult addEntity(String name, String value) {
        if (entities == null) {
            entities = new HashMap<>();
        }
        entities.put(name, value);
        return this;
    }
}
