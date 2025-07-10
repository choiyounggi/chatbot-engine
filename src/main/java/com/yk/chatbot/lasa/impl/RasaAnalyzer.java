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
                        // 특별시/광역시
                        "서울", "부산", "인천", "대구", "광주", "대전", "울산", "세종",
                        "서울특별시", "부산광역시", "인천광역시", "대구광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
                        
                        // 도
                        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
                        "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주도", "제주특별자치도",
                        
                        // 주요 도시
                        "수원", "성남", "안양", "안산", "고양", "용인", "청주", "천안", "전주", "포항", "창원",
                        "김해", "평택", "강릉", "원주", "춘천", "속초", "여수", "순천", "목포", "경주", "구미",
                        "거제", "양산", "진주", "파주", "의정부", "남양주", "화성", "시흥", "광명", "하남", "군포",
                        "오산", "이천", "안성", "광주", "파주", "김포", "구리", "여주", "양주", "동두천", "과천",
                        "의왕", "포천", "양평", "동해", "태백", "삼척", "정선", "홍천", "횡성", "영월", "평창",
                        "정읍", "남원", "김제", "익산", "완주", "진안", "무주", "장수", "임실", "순창", "고창", "부안",
                        "나주", "광양", "담양", "곡성", "구례", "고흥", "보성", "화순", "장흥", "강진", "해남",
                        "영암", "무안", "함평", "영광", "장성", "완도", "진도", "신안",
                        "영덕", "울진", "문경", "예천", "안동", "영양", "영주", "봉화", "울릉", "의성", "청송", "영천",
                        "경산", "청도", "고령", "성주", "칠곡", "김천", "군위", "사천", "밀양", "의령", "함안",
                        "창녕", "고성", "남해", "하동", "산청", "함양", "거창", "합천", "통영"
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
