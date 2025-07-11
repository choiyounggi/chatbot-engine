package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.AnalysisResult;
import com.yk.chatbot.service.WeatherService;
import kr.co.shineware.nlp.komoran.constant.DEFAULT_MODEL;
import kr.co.shineware.nlp.komoran.core.Komoran;
import kr.co.shineware.nlp.komoran.model.KomoranResult;
import kr.co.shineware.nlp.komoran.model.Token;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * KOMORAN을 활용한 한국어 텍스트 분석기
 */
@Slf4j
@Component
public class KoalaNLPAnalyzer implements Analyze {

    private final Komoran komoran;
    private final Map<String, Set<String>> intentKeywords = initIntentKeywords();
    private final Set<String> cityNames;
    private final WeatherService weatherService;

    @Autowired
    public KoalaNLPAnalyzer(WeatherService weatherService) {
        this.weatherService = weatherService;
        this.cityNames = new HashSet<>(weatherService.getCityCoordinates().keySet());
        
        try {
            // KOMORAN 분석기 초기화 (FULL 모델 사용)
            komoran = new Komoran(DEFAULT_MODEL.FULL);
            initDictionaries();
            log.info("KOMORAN 분석기/사전 초기화 완료");
        } catch (Exception e) {
            log.error("KOMORAN 분석기 초기화 실패", e);
            throw new RuntimeException("KOMORAN 초기화 실패", e);
        }
    }

    private void initDictionaries() {
        try {
            // 사용자 사전 항목 생성 - WeatherService의 모든 도시 사용
            Set<String> locations = cityNames;
            
            // 제주도 추가 (제주만 있는 경우)
            if (locations.contains("제주")) {
                locations.add("제주도");
            }
            
            // 임시 파일에 사용자 사전 생성
            File tempFile = File.createTempFile("komoran_user_dic", ".txt");
            BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile));
            
            for (String location : locations) {
                writer.write(location + "\tNNP");
                writer.newLine();
            }
            writer.close();
            
            // 사용자 사전 파일 경로 설정
            komoran.setUserDic(tempFile.getAbsolutePath());
            
            log.debug("KOMORAN 사용자 사전에 {} 항목 추가됨 ({})", locations.size(), tempFile.getAbsolutePath());
        } catch (Exception e) {
            log.warn("KOMORAN 사용자 사전 추가 중 오류 발생", e);
        }
    }

    private static Map<String, Set<String>> initIntentKeywords() {
        Map<String, Set<String>> keywords = new HashMap<>();
        keywords.put("weather", Set.of("날씨", "기온", "비", "눈", "맑음", "흐림", "습도", "미세먼지", "예보"));
        keywords.put("greeting", Set.of("안녕", "반가워", "안녕하세요", "하이", "헬로"));
        return keywords;
    }

    /**
     * 단일 파라미터 분석 메서드 - Analyze 인터페이스 구현
     */
    @Override
    public AnalysisResult analyze(String message) {
        return analyze(message, null);
    }

    @Override
    public AnalysisResult analyze(String message, String userId) {
        try {
            log.info("메시지 분석 시작: {}", message);

            KomoranResult komoranResult = komoran.analyze(message);
            String intent = determineIntent(komoranResult);
            double confidence = calcConfidence(komoranResult, intent);

            AnalysisResult result = AnalysisResult.builder()
                    .intent(intent)
                    .confidence(confidence)
                    .originalMessage(message)
                    .build();

            if ("weather".equals(intent)) {
                extractWeatherEntities(komoranResult, result);
            }

            log.info("분석 완료: intent={}, confidence={}, entities={}", result.getIntent(), result.getConfidence(), result.getEntities());
            return result;
        } catch (Exception e) {
            log.error("메시지 분석 중 오류: {}", message, e);
            return AnalysisResult.builder()
                    .intent("error")
                    .confidence(0.0)
                    .originalMessage(message)
                    .build();
        }
    }

    /** 의도 결정: 키워드 매칭 스코어 기반 */
    private String determineIntent(KomoranResult komoranResult) {
        Map<String, Integer> intentScores = new HashMap<>();

        List<Token> tokens = komoranResult.getTokenList();
        for (Token token : tokens) {
            String morphText = token.getMorph().toLowerCase();
            String pos = token.getPos();
            
            // 명사, 동사, 형용사 등 주요 품사에 대해 매칭
            if (pos.startsWith("NN") || pos.startsWith("VV") || pos.startsWith("VA")) {
                for (Map.Entry<String, Set<String>> entry : intentKeywords.entrySet()) {
                    if (entry.getValue().contains(morphText)) {
                        intentScores.put(entry.getKey(), intentScores.getOrDefault(entry.getKey(), 0) + 1);
                    }
                }
            }
        }

        return intentScores.entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("fallback");
    }

    /** 신뢰도: intent별 키워드 매칭 수 기반 동적 계산 */
    private double calcConfidence(KomoranResult komoranResult, String intent) {
        if ("fallback".equals(intent) || "error".equals(intent)) return 0.0;
        
        Set<String> keywords = intentKeywords.getOrDefault(intent, Set.of());
        List<Token> tokens = komoranResult.getTokenList();
        
        long matched = tokens.stream()
                .filter(token -> keywords.contains(token.getMorph().toLowerCase()))
                .count();
                
        return Math.min(1.0, 0.5 + 0.15 * matched);
    }

    /** 날씨 관련 개체(주로 위치) 추출 */
    private void extractWeatherEntities(KomoranResult komoranResult, AnalysisResult result) {
        List<String> locations = new ArrayList<>();
        List<Token> tokens = komoranResult.getTokenList();
        
        for (Token token : tokens) {
            // 고유명사(NNP)인 경우 위치 확인
            if ("NNP".equals(token.getPos())) {
                String loc = token.getMorph();
                if (isLocationName(loc)) {
                    locations.add(loc);
                }
            }
        }
        
        // 추출된 위치가 없으면 기본값(서울) 사용
        result.addEntity("location", locations.isEmpty() ? "서울" : String.join(",", locations));
    }

    private boolean isLocationName(String text) {
        // WeatherService의 도시 목록을 기반으로 체크
        if (cityNames.contains(text)) {
            return true;
        }
        
        // 특별한 케이스: 제주/제주도
        if ("제주도".equals(text) && cityNames.contains("제주")) {
            return true;
        }
        
        return false;
    }
}
