package com.yk.chatbot.lasa;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.HashMap;
import java.util.Map;

/**
 * 문제 해결 결과 클래스
 * 분석된 의도에 따른 비즈니스 로직 수행 결과를 저장합니다.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SolutionResult {
    
    /**
     * 처리 상태 (success, error, etc.)
     */
    private String status;
    
    /**
     * 응답 생성에 필요한 데이터
     */
    @Builder.Default
    private Map<String, Object> data = new HashMap<>();
    
    /**
     * 응답 메시지 템플릿
     */
    private String responseTemplate;
    
    /**
     * 원본 의도
     */
    private String originalIntent;
    
    /**
     * 상태가 성공인지 확인
     */
    public boolean isSuccess() {
        return "success".equals(status);
    }
    
    /**
     * 데이터 추가 헬퍼 메소드
     */
    public SolutionResult addData(String key, Object value) {
        if (data == null) {
            data = new HashMap<>();
        }
        data.put(key, value);
        return this;
    }
}
