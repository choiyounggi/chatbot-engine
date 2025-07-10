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
            if (entities != null && !entities.isEmpty()) {
                log.info("엔티티 처리 시작 (총 {}개)", entities.size());
                entities.forEach((key, value) -> {
                    if (value != null) {
                        String entityValue = value.toString().trim();
                        if (!entityValue.isEmpty()) {
                            result.addEntity(key, entityValue);
                            log.info("엔티티 추가: {}={}", key, entityValue);
                        } else {
                            log.warn("빈 엔티티 값 무시: {}", key);
                        }
                    } else {
                        log.warn("null 엔티티 값 무시: {}", key);
                    }
                });
                
                // location 엔티티 특별 처리 (KoalaNLP 활용)
                if (!result.hasEntity("location") && intent.equals("weather") || intent.equals("temperature")) {
                    log.info("location 엔티티를 찾을 수 없어 직접 추출 시도");
                    // KoalaNLP 또는 다른 방법으로 위치 정보 직접 추출 가능
                    // 여기서는 간단한 키워드 매칭으로 구현
                    String lowerMessage = message.toLowerCase();
                    
                    // 주요 도시/지역 키워드
                    String[] locationKeywords = {
                        "서울", "부산", "인천", "대구", "광주", "대전", "울산", "세종",
                        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
                        "수원", "성남", "안양", "안산", "고양", "용인", "청주", "천안", "전주", "포항", "창원"
                    };
                    
                    for (String keyword : locationKeywords) {
                        if (lowerMessage.contains(keyword.toLowerCase())) {
                            result.addEntity("location", keyword);
                            log.info("직접 위치 엔티티 추출 성공: {}", keyword);
                            break;
                        }
                    }
                }
            } else {
                log.warn("추출된 엔티티가 없습니다");
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
