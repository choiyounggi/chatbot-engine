package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.AnalysisResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

/**
 * Rasa와 KOMORAN을 조합한 하이브리드 한국어 분석기
 * 두 분석기의 장점을 활용하여 정확도를 높입니다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class HybridAnalyzer implements Analyze {

    private final RasaAnalyzer rasaAnalyzer;
    private final KoalaNLPAnalyzer koalaNLPAnalyzer; // KOMORAN 기반 분석기
    
    // 의도 신뢰도 임계값 (이 값 이상이면 높은 신뢰도로 간주)
    private static final double HIGH_CONFIDENCE_THRESHOLD = 0.7;
    
    @Override
    public AnalysisResult analyze(String message) {
        return analyze(message, null);
    }
    
    @Override
    public AnalysisResult analyze(String message, String userId) {
        try {
            log.info("하이브리드 분석 시작: {}", message);
            
            // 두 분석기로 각각 분석 실행
            AnalysisResult rasaResult = rasaAnalyzer.analyze(message, userId);
            AnalysisResult koalaNLPResult = koalaNLPAnalyzer.analyze(message, userId);
            
            // 최종 결과를 생성할 빌더
            AnalysisResult result = AnalysisResult.builder()
                    .originalMessage(message)
                    .build();
            
            // 의도 결정 로직
            determineIntent(rasaResult, koalaNLPResult, result);
            
            // 엔티티 통합 로직
            mergeEntities(rasaResult, koalaNLPResult, result);
            
            log.info("하이브리드 분석 완료: 의도={}, 신뢰도={}, 엔티티 개수={}", 
                    result.getIntent(), result.getConfidence(), 
                    result.getEntities() != null ? result.getEntities().size() : 0);
            
            return result;
        } catch (Exception e) {
            log.error("하이브리드 분석 중 오류 발생", e);
            return AnalysisResult.builder()
                    .intent("error")
                    .confidence(0.0)
                    .originalMessage(message)
                    .build();
        }
    }
    
    /**
     * 두 분석기의 결과를 바탕으로 최종 의도 결정
     * 높은 신뢰도를 가진 분석기의 의도를 우선시합니다.
     */
    private void determineIntent(AnalysisResult rasaResult, AnalysisResult koalaNLPResult, 
                                AnalysisResult result) {
        String finalIntent;
        double finalConfidence;
        
        // 두 분석기의 의도가 같으면 해당 의도를 사용하고 신뢰도를 높임
        if (rasaResult.getIntent().equals(koalaNLPResult.getIntent())) {
            finalIntent = rasaResult.getIntent();
            // 두 신뢰도의 평균에 보너스 추가 (최대 1.0)
            finalConfidence = Math.min(1.0, 
                (rasaResult.getConfidence() + koalaNLPResult.getConfidence()) / 2 + 0.1);
        }
        // 의도가 다른 경우 신뢰도가 높은 쪽을 선택
        else if (rasaResult.getConfidence() >= HIGH_CONFIDENCE_THRESHOLD && 
                 rasaResult.getConfidence() > koalaNLPResult.getConfidence()) {
            finalIntent = rasaResult.getIntent();
            finalConfidence = rasaResult.getConfidence();
        } 
        else if (koalaNLPResult.getConfidence() >= HIGH_CONFIDENCE_THRESHOLD && 
                 koalaNLPResult.getConfidence() > rasaResult.getConfidence()) {
            finalIntent = koalaNLPResult.getIntent();
            finalConfidence = koalaNLPResult.getConfidence();
        }
        // 둘 다 신뢰도가 낮으면 Rasa 결과를 기본으로 사용하되 신뢰도 감소
        else {
            finalIntent = rasaResult.getIntent();
            finalConfidence = rasaResult.getConfidence() * 0.9; // 신뢰도 감소
        }
        
        // 의도와 신뢰도 설정
        result.setIntent(finalIntent);
        result.setConfidence(finalConfidence);
    }
    
    /**
     * 두 분석기에서 추출한 엔티티를 통합
     * 중복된 엔티티 유형이 있을 경우 처리 로직 포함
     */
    private void mergeEntities(AnalysisResult rasaResult, AnalysisResult koalaNLPResult, 
                              AnalysisResult result) {
        Map<String, String> mergedEntities = new HashMap<>();
        
        // Rasa 엔티티 추가
        if (rasaResult.getEntities() != null) {
            mergedEntities.putAll(rasaResult.getEntities());
        }
        
        // KOMORAN 엔티티 추가/병합
        if (koalaNLPResult.getEntities() != null) {
            for (Map.Entry<String, String> entry : koalaNLPResult.getEntities().entrySet()) {
                String key = entry.getKey();
                String koalaNLPValue = entry.getValue();
                
                // 이미 해당 엔티티 유형이 있는 경우
                if (mergedEntities.containsKey(key)) {
                    String rasaValue = mergedEntities.get(key);
                    
                    // 값이 같으면 그대로 유지
                    if (koalaNLPValue.equals(rasaValue)) {
                        continue;
                    }
                    
                    // 값이 다르면 처리 전략
                    // 1. 엔티티 유형별 우선순위 적용
                    if (key.equals("location")) {
                        // 위치 엔티티는 KOMORAN이 더 정확할 가능성이 높음
                        mergedEntities.put(key, koalaNLPValue);
                        log.debug("위치 엔티티 충돌 해결: KOMORAN 우선 적용 ({})", koalaNLPValue);
                    } 
                    else if (key.equals("datetime")) {
                        // 날짜/시간 엔티티는 KOMORAN이 더 정확할 가능성이 높음
                        mergedEntities.put(key, koalaNLPValue);
                        log.debug("시간 엔티티 충돌 해결: KOMORAN 우선 적용 ({})", koalaNLPValue);
                    }
                    else if (key.equals("person")) {
                        // 인물 엔티티는 KOMORAN이 더 정확할 가능성이 높음
                        mergedEntities.put(key, koalaNLPValue);
                    }
                    // 2. 기타 엔티티는 기본적으로 Rasa 유지 (이미 설정되어 있음)
                } 
                // 새로운 엔티티 유형이면 추가
                else {
                    mergedEntities.put(key, koalaNLPValue);
                }
            }
        }
        
        // 기존 엔티티가 있으면 초기화
        result.setEntities(new HashMap<>());
        
        // 병합된 엔티티 추가
        for (Map.Entry<String, String> entry : mergedEntities.entrySet()) {
            result.addEntity(entry.getKey(), entry.getValue());
        }
        
        // 주요 엔티티 로깅
        if (!mergedEntities.isEmpty()) {
            log.info("병합된 엔티티: {}", mergedEntities);
        }
    }
}
