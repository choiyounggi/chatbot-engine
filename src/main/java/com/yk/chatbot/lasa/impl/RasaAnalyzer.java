package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.service.RasaClientService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Rasa NLU를 활용한 한국어 분석기 구현체
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RasaAnalyzer implements Analyze {

    private final RasaClientService rasaClientService;
    
    @Override
    public AnalysisResult analyze(String message) {
        return analyze(message, null);
    }
    
    @Override
    public AnalysisResult analyze(String message, String userId) {
        try {
            log.info("메시지 분석 시작: {}", message);
            
            // Rasa NLU 서비스 호출
            Map<String, Object> rasaResult = rasaClientService.parseMessage(message, userId);
            
            String intent = (String) rasaResult.get("intent");
            double confidence = (double) rasaResult.get("confidence");
            
            // 분석 결과 생성
            AnalysisResult result = AnalysisResult.builder()
                    .intent(intent)
                    .confidence(confidence)
                    .originalMessage(message)
                    .build();
            
            // 엔티티 추가
            @SuppressWarnings("unchecked")
            Map<String, Object> entities = (Map<String, Object>) rasaResult.get("entities");
            if (entities != null) {
                entities.forEach((key, value) -> {
                    result.addEntity(key, value.toString());
                });
            }
            
            log.info("메시지 분석 완료: 의도={}, 신뢰도={}, 엔티티 개수={}", 
                    result.getIntent(), result.getConfidence(), 
                    result.getEntities() != null ? result.getEntities().size() : 0);
            
            return result;
        } catch (Exception e) {
            log.error("메시지 분석 중 오류 발생", e);
            return AnalysisResult.builder()
                    .intent("error")
                    .confidence(0.0)
                    .originalMessage(message)
                    .build();
        }
    }
}
