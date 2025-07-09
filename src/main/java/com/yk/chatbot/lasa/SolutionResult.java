package com.yk.chatbot.lasa;

import lombok.Builder;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

/**
 * 문제 해결 결과를 담는 클래스입니다.
 * 해결 상태, 결과 데이터, 응답 템플릿 등을 포함합니다.
 */
@Getter
@Builder
public class SolutionResult {
    /**
     * 해결 상태 (성공, 실패, 부분 성공 등)
     */
    private String status;
    
    /**
     * 응답에 사용할 데이터
     */
    @Builder.Default
    private Map<String, Object> data = new HashMap<>();
    
    /**
     * 응답 메시지 템플릿
     */
    private String responseTemplate;
    
    /**
     * 원본 의도 (분석 단계에서 추출된 의도)
     */
    private String originalIntent;
    
    /**
     * 추가 데이터를 저장하는 헬퍼 메서드
     * @param key 데이터 키
     * @param value 데이터 값
     */
    public void addData(String key, Object value) {
        data.put(key, value);
    }
    
    /**
     * 해결이 성공적으로 완료되었는지 확인
     * @return 성공 여부
     */
    public boolean isSuccess() {
        return "success".equals(status);
    }
}
