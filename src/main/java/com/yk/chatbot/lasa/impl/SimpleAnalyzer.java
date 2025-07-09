package com.yk.chatbot.lasa.impl;

import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.AnalysisResult;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Analyze 인터페이스의 기본 구현체입니다.
 * 규칙 기반으로 사용자 의도와 엔티티를 추출합니다.
 */
@Component
public class SimpleAnalyzer implements Analyze {

    // 의도 및 엔티티 추출을 위한 패턴
    private static final Map<String, Pattern> intentPatterns = new HashMap<>();
    
    // 정규식 패턴 초기화
    static {
        // 인사 의도
        intentPatterns.put("greeting", Pattern.compile("(안녕|반가워|헬로|하이|hi|hello)", Pattern.CASE_INSENSITIVE));
        
        // 날씨 조회 의도
        intentPatterns.put("weather", Pattern.compile("날씨.*(어때|어떤가|어떄|알려|궁금|조회|확인)", Pattern.CASE_INSENSITIVE));
        
        // 시간 조회 의도
        intentPatterns.put("time_query", Pattern.compile("(시간|몇시).*?(알려|궁금|지금|현재)", Pattern.CASE_INSENSITIVE));
        
        // 도움말 의도
        intentPatterns.put("help", Pattern.compile("(도움|도와|명령어|기능|할수있|할 수 있|가능한).*?(뭐|무엇|알려|뭘|모두|전부)", Pattern.CASE_INSENSITIVE));
    }
    
    @Override
    public AnalysisResult analyze(String message) {
        if (message == null || message.isEmpty()) {
            return AnalysisResult.builder()
                    .intent("unknown")
                    .confidence(0.0)
                    .build();
        }
        
        // 가장 높은 신뢰도를 가진 의도 찾기
        String matchedIntent = "unknown";
        double highestConfidence = 0.0;
        
        for (Map.Entry<String, Pattern> entry : intentPatterns.entrySet()) {
            Matcher matcher = entry.getValue().matcher(message);
            if (matcher.find()) {
                // 단순한 구현에서는 첫 번째 매치되는 의도를 반환
                matchedIntent = entry.getKey();
                highestConfidence = 1.0; // 정규식 매칭은 이진적이므로 1.0으로 설정
                break;
            }
        }
        
        // 엔티티 추출 (이 예제에서는 간단한 구현)
        Map<String, String> entities = extractEntities(message, matchedIntent);
        
        return AnalysisResult.builder()
                .intent(matchedIntent)
                .entities(entities)
                .confidence(highestConfidence)
                .build();
    }
    
    /**
     * 의도에 따라 메시지에서 엔티티를 추출합니다.
     * @param message 사용자 메시지
     * @param intent 분석된 의도
     * @return 추출된 엔티티 맵
     */
    private Map<String, String> extractEntities(String message, String intent) {
        Map<String, String> entities = new HashMap<>();
        
        switch (intent) {
            case "weather":
                // 지역 엔티티 추출 예시
                Pattern locationPattern = Pattern.compile("(서울|부산|대구|인천|광주|대전|울산|제주)");
                Matcher locationMatcher = locationPattern.matcher(message);
                if (locationMatcher.find()) {
                    entities.put("location", locationMatcher.group(1));
                }
                break;
            case "time_query":
                // 시간 관련 엔티티 추출 (있다면)
                Pattern timePattern = Pattern.compile("(오전|오후|저녁|아침)");
                Matcher timeMatcher = timePattern.matcher(message);
                if (timeMatcher.find()) {
                    entities.put("time_period", timeMatcher.group(1));
                }
                break;
        }
        
        return entities;
    }
}
