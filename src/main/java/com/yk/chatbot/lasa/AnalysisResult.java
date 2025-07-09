package com.yk.chatbot.lasa;

import lombok.Builder;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

/**
 * 사용자 메시지 분석 결과를 담는 클래스입니다.
 * 의도(intent)와 추출된 엔티티 정보를 포함합니다.
 */
@Getter
@Builder
public class AnalysisResult {
    /**
     * 사용자 메시지에서 파악된 의도
     */
    private String intent;
    
    /**
     * 사용자 메시지에서 추출한 엔티티들
     * 키는 엔티티 타입, 값은 엔티티 값입니다.
     */
    @Builder.Default
    private Map<String, String> entities = new HashMap<>();
    
    /**
     * 분석 신뢰도 점수 (0.0 ~ 1.0)
     */
    private double confidence;
    
    /**
     * 엔티티를 추가하는 헬퍼 메서드
     * @param type 엔티티 타입
     * @param value 엔티티 값
     */
    public void addEntity(String type, String value) {
        entities.put(type, value);
    }
}
