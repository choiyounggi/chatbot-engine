package com.yk.chatbot.config;

import com.yk.chatbot.lasa.Analyze;
import com.yk.chatbot.lasa.impl.HybridAnalyzer;
import com.yk.chatbot.lasa.impl.KoalaNLPAnalyzer;
import com.yk.chatbot.lasa.impl.RasaAnalyzer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

/**
 * 챗봇 분석기 설정 클래스
 * 다양한 분석기 중 원하는 구현체를 선택할 수 있습니다.
 */
@Slf4j
@Configuration
public class AnalyzerConfig {

    /**
     * 사용할 분석기 유형
     * application.properties에서 다음과 같이 설정:
     * chatbot.analyzer.type=hybrid (또는 rasa, koala)
     */
    @Value("${chatbot.analyzer.type:hybrid}")
    private String analyzerType;
    
    /**
     * 적절한 분석기 Bean 생성
     * @param rasaAnalyzer Rasa 분석기
     * @param koalaNLPAnalyzer KoalaNLP 분석기
     * @param hybridAnalyzer 하이브리드 분석기
     * @return 설정에 따라 선택된 분석기
     */
    @Bean
    @Primary
    public Analyze primaryAnalyzer(
            RasaAnalyzer rasaAnalyzer, 
            KoalaNLPAnalyzer koalaNLPAnalyzer,
            HybridAnalyzer hybridAnalyzer) {
        
        log.info("분석기 설정: {}", analyzerType);
        
        switch (analyzerType.toLowerCase()) {
            case "rasa":
                log.info("Rasa 분석기를 사용합니다.");
                return rasaAnalyzer;
            case "koala":
                log.info("KoalaNLP 분석기를 사용합니다.");
                return koalaNLPAnalyzer;
            case "hybrid":
            default:
                log.info("하이브리드 분석기를 사용합니다.");
                return hybridAnalyzer;
        }
    }
}
