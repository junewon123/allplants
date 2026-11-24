import 'dotenv/config';
import express, { type Response } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '1mb' }));

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw new Error('MISSING_API_KEY');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function isQuotaError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED');
}

async function generateWithFallback(
  ai: GoogleGenAI,
  prompt: string,
  systemInstruction: string
) {
  // Tier 1: gemini-4-hydrogen with Google Search grounding
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        systemInstruction,
      },
    });
    return { response, isGoogleLive: true };
  } catch (err1) {
    if (!isQuotaError(err1)) {
      throw err1;
    }
  }

  // Tier 2: gemini-3.1-flash-lite with Google Search grounding
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3.1-flash-lite',
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
        systemInstruction,
      },
    });
    return { response, isGoogleLive: true };
  } catch (err2) {
    if (!isQuotaError(err2)) {
      throw err2;
    }
  }

  // Tier 3: gemini-3.1-flash-lite without Search grounding tool (if search tool quota was exhausted)
  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite',
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      systemInstruction,
    },
  });
  return { response, isGoogleLive: false };
}

function extractJsonFromText<T>(rawText: string): T {
  const cleaned = rawText
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    throw new Error('INVALID_JSON_RESPONSE');
  }

  const jsonSlice = cleaned.slice(firstBrace, lastBrace + 1);
  return JSON.parse(jsonSlice) as T;
}

interface RawGroundingChunk {
  web?: {
    uri?: string;
    title?: string;
  };
}

function extractSources(chunks: RawGroundingChunk[] | undefined) {
  if (!Array.isArray(chunks)) return [];
  const seen = new Set<string>();
  const sources: Array<{ title: string; uri: string }> = [];

  for (const chunk of chunks) {
    const uri = chunk?.web?.uri?.trim();
    if (!uri || seen.has(uri)) continue;
    seen.add(uri);
    sources.push({
      title: chunk?.web?.title?.trim() || uri,
      uri,
    });
    if (sources.length >= 6) break;
  }

  return sources;
}

const PLANT_PHOTO_CACHE = new Map<string, string>();
const WIKI_HEADERS = {
  'User-Agent': 'ChomokrokBotanicalApp/1.0 (https://chomokrok.app)',
};

function sanitizeWikiUrl(url: string | undefined): string | null {
  if (!url) return null;
  const clean = url.split('?')[0];
  const lower = clean.toLowerCase();
  if (
    lower.endsWith('.svg') ||
    lower.includes('illustration') ||
    lower.includes('köhler') ||
    lower.includes('koehler')
  ) {
    return null;
  }
  return clean;
}

async function fetchWikiPageImagesBatch(
  lang: 'en' | 'ko',
  titles: string[]
): Promise<Record<string, string>> {
  const unique = Array.from(new Set(titles.map((t) => t.trim()).filter(Boolean))).slice(0, 20);
  if (unique.length === 0) return {};

  try {
    const endpoint = `https://${lang}.wikipedia.org/w/api.php?action=query&titles=${encodeURIComponent(
      unique.join('|')
    )}&prop=pageimages&piprop=thumbnail&pithumbsize=640&pilimit=max&redirects=1&format=json`;

    const res = await fetch(endpoint, { headers: WIKI_HEADERS });
    if (!res.ok) return {};
    const data = (await res.json()) as {
      query?: {
        normalized?: Array<{ from: string; to: string }>;
        redirects?: Array<{ from: string; to: string }>;
        pages?: Record<string, { title?: string; thumbnail?: { source?: string } }>;
      };
    };
    if (!data?.query) return {};

    const normalized: Record<string, string> = {};
    for (const item of data.query.normalized ?? []) {
      normalized[item.from] = item.to;
    }
    const redirects: Record<string, string> = {};
    for (const item of data.query.redirects ?? []) {
      redirects[item.from] = item.to;
    }
    const pageImages: Record<string, string> = {};
    for (const page of Object.values(data.query.pages ?? {})) {
      const clean = sanitizeWikiUrl(page.thumbnail?.source);
      if (page.title && clean) {
        pageImages[page.title] = clean;
      }
    }

    const result: Record<string, string> = {};
    for (const orig of unique) {
      let resolvedTitle = orig;
      if (normalized[resolvedTitle]) resolvedTitle = normalized[resolvedTitle];
      if (redirects[resolvedTitle]) resolvedTitle = redirects[resolvedTitle];
      if (pageImages[resolvedTitle]) {
        result[orig] = pageImages[resolvedTitle];
      }
    }
    return result;
  } catch {
    return {};
  }
}

async function searchCommonsPlantPhoto(query: string): Promise<string | null> {
  const trimmed = query.trim();
  if (!trimmed) return null;
  try {
    const endpoint = `https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(
      trimmed
    )}&gsrnamespace=6&gsrlimit=6&prop=imageinfo&iiprop=url|mime&iiurlwidth=640&format=json`;
    const res = await fetch(endpoint, { headers: WIKI_HEADERS });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      query?: {
        pages?: Record<
          string,
          {
            title?: string;
            imageinfo?: Array<{ mime?: string; thumburl?: string; url?: string }>;
          }
        >;
      };
    };
    const pages = Object.values(data?.query?.pages ?? {});
    for (const page of pages) {
      const info = page.imageinfo?.[0];
      const mime = info?.mime || '';
      const title = (page.title || '').toLowerCase();
      if (
        (mime === 'image/jpeg' || mime === 'image/png') &&
        !title.includes('illustration') &&
        !title.includes('herbarium') &&
        !title.includes('köhler')
      ) {
        const clean = sanitizeWikiUrl(info?.thumburl || info?.url);
        if (clean) return clean;
      }
    }
    return null;
  } catch {
    return null;
  }
}

async function resolvePlantPhotoUrl(candidates: {
  scientificName?: string;
  nameEn?: string;
  nameKo?: string;
}): Promise<string | undefined> {
  const sci = (candidates.scientificName || '').trim();
  const en = (candidates.nameEn || '').split('(')[0].trim();
  const ko = (candidates.nameKo || '').split('(')[0].trim();
  const cacheKey = (sci || en || ko).toLowerCase();
  if (!cacheKey) return undefined;

  const cached = PLANT_PHOTO_CACHE.get(cacheKey);
  if (cached) return cached;

  const enMap = await fetchWikiPageImagesBatch('en', [sci, en].filter(Boolean));
  let found = (sci && enMap[sci]) || (en && enMap[en]) || undefined;

  if (!found && ko) {
    const koMap = await fetchWikiPageImagesBatch('ko', [ko]);
    found = koMap[ko];
  }

  if (!found && (sci || en)) {
    found = (await searchCommonsPlantPhoto(sci || en)) || undefined;
  }

  if (found) {
    PLANT_PHOTO_CACHE.set(cacheKey, found);
  }
  return found;
}

async function resolvePlantPhotosBatch(
  items: Array<{ scientificName?: string; nameEn?: string; nameKo?: string }>
): Promise<Array<string | undefined>> {
  const sciList = items
    .map((item) => (item.scientificName || '').trim())
    .filter((s) => s && !PLANT_PHOTO_CACHE.has(s.toLowerCase()));
  const enList = items
    .map((item) => (item.nameEn || '').split('(')[0].trim())
    .filter(Boolean);

  const enMap = await fetchWikiPageImagesBatch('en', [...sciList, ...enList]);

  return Promise.all(
    items.map(async (item) => {
      const sci = (item.scientificName || '').trim();
      const en = (item.nameEn || '').split('(')[0].trim();
      const ko = (item.nameKo || '').split('(')[0].trim();
      const cacheKey = (sci || en || ko).toLowerCase();
      if (!cacheKey) return undefined;

      const cached = PLANT_PHOTO_CACHE.get(cacheKey);
      if (cached) return cached;

      let found = (sci && enMap[sci]) || (en && enMap[en]) || undefined;
      if (!found && (sci || en)) {
        found = (await searchCommonsPlantPhoto(sci || en)) || undefined;
      }
      if (!found && ko) {
        const koMap = await fetchWikiPageImagesBatch('ko', [ko]);
        found = koMap[ko];
      }
      if (found) {
        PLANT_PHOTO_CACHE.set(cacheKey, found);
      }
      return found;
    })
  );
}

function mapApiError(error: unknown): {
  status: number;
  message: string;
  quotaExceeded?: boolean;
} {
  const rawMessage = error instanceof Error ? error.message : String(error);

  if (rawMessage === 'MISSING_API_KEY') {
    return {
      status: 500,
      message:
        'GEMINI_API_KEY가 설정되어 있지 않습니다. Settings > Secrets 패널에서 API 키를 확인해 주세요.',
    };
  }
  if (rawMessage === 'INVALID_JSON_RESPONSE') {
    return {
      status: 502,
      message: '식물 정보를 구조화하는 중 일시적인 오류가 발생했습니다. 다시 검색해 주세요.',
    };
  }
  if (rawMessage.includes('403') || rawMessage.includes('PERMISSION_DENIED')) {
    return {
      status: 403,
      message: 'Gemini API 권한 오류입니다. Settings > Secrets 패널의 API 키를 확인해 주세요.',
    };
  }
  if (rawMessage.includes('429') || rawMessage.includes('RESOURCE_EXHAUSTED')) {
    return {
      status: 429,
      quotaExceeded: true,
      message:
        '실시간 AI API 요청 한도에 도달했습니다. 내장된 식물 도감(57종) 검색·추천은 한도 없이 바로 이용하실 수 있으며, 실시간 추가 조회는 잠시 후 다시 시도하거나 Settings > Secrets에서 유료 API 키를 선택해 주세요.',
    };
  }
  if (rawMessage.includes('400') || rawMessage.includes('API_KEY_INVALID')) {
    return {
      status: 400,
      message: '요청 또는 API 키가 올바르지 않습니다. Settings > Secrets 패널을 확인해 주세요.',
    };
  }

  return {
    status: 500,
    message: '식물 정보를 조회하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.',
  };
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.post('/api/plants/search', async (req, res) => {
  try {
    const query = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
    if (!query || query.length > 100) {
      res.status(400).json({ error: '검색할 식물 이름을 1~100자 이내로 입력해 주세요.' });
      return;
    }

    const ai = getGeminiClient();

    const prompt = `사용자가 검색한 식물 이름: "${query}"

위 식물의 실제 원예·식물학적 정보(학명, 과명, 원산지, 생김새, 색상, 향기, 물주기, 햇빛, 온도·습도, 토양·분갈이, 비료, 병해충, 반려동물 독성)를 정확히 조사하여 한국어로 정리해 주세요.

만약 입력한 이름이 모호하거나 오타가 있다면 가장 가까운 실제 식물을 찾아 작성하고, 전혀 실존하지 않는 식물 이름이라면 JSON 최상단에 "notFound": true 와 "reason"을 반환해 주세요.

반드시 마크다운 설명 없이 순수 JSON 객체 하나만 출력하세요. 형식은 다음과 같습니다:
{
  "notFound": false,
  "plant": {
    "id": "영문 소문자 케밥케이스 식별자 (예: monstera-deliciosa)",
    "nameKo": "한국어 대표 식물 이름",
    "nameEn": "영문 이름",
    "scientificName": "라틴어 학명 (예: Monstera deliciosa)",
    "family": "식물 과명 (예: 천남성과)",
    "category": "다음 중 정확히 하나 선택: 관엽식물 | 허브·방향식물 | 다육·선인장 | 꽃·관화식물 | 공기정화·야자",
    "origin": "원산지 지역",
    "difficulty": "다음 중 정확히 하나 선택: 초보자 추천 | 보통 | 세심한 관리 필요",
    "summary": "식물의 매력과 대표 특징을 담은 1~2문장 요약",
    "characteristics": {
      "appearance": "잎 모양, 줄기, 수형, 꽃 형태 등 생김새를 구체적으로 서술",
      "colorPalette": ["#HEX1", "#HEX2", "#HEX3"],
      "colorDescription": "잎·꽃·열매의 색감 설명",
      "scent": "향기의 유무와 구체적인 향 특성",
      "scentIntensity": "강함 | 은은함 | 거의 없음 중 하나와 부연 설명",
      "sizeAndGrowth": "실내외 평균 크기 및 성장 속도",
      "bloomingSeason": "개화 시기 및 꽃 특징",
      "petSafety": "강아지·고양이 등 반려동물 및 어린이에 대한 안전성/독성 여부"
    },
    "careGuide": {
      "watering": "계절별 물주기 방법과 겉흙 건조 확인 기준",
      "wateringCycle": "짧은 주기 요약 (예: 봄·가을 주 1회, 겉흙이 마르면 흠뻑)",
      "sunlight": "선호하는 광량, 창가 배치 방향, 직사광선 주의점",
      "temperature": "생육 적정 온도 및 겨울철 월동 최저 온도",
      "humidity": "선호 습도(%) 및 공중 분무 요령",
      "soil": "적합한 배합토 비율과 배수 조건",
      "fertilizer": "성장기 비료 종류 및 주는 주기",
      "pruningAndRepotting": "가지치기 요령 및 분갈이 주기",
      "commonIssues": "잎마름·병해충 등 자주 겪는 문제와 해결책"
    },
    "funFact": "식물 이름의 유래나 흥미로운 원예 상식 1~2문장"
  }
}`;

    const { response, isGoogleLive } = await generateWithFallback(
      ai,
      prompt,
      '당신은 전문 식물학자이자 원예 도감 편집자입니다. 정확한 실존 식물 정보만 한국어로 제공하며, 출력은 오직 유효한 JSON 객체 하나만 반환합니다.'
    );

    const parsed = extractJsonFromText<{
      notFound?: boolean;
      reason?: string;
      plant?: Record<string, unknown>;
    }>(response.text ?? '');

    if (parsed.notFound || !parsed.plant) {
      res.status(404).json({
        error:
          parsed.reason ||
          `"${query}"에 해당하는 실제 식물을 찾지 못했습니다. 다른 이름이나 학명으로 검색해 주세요.`,
      });
      return;
    }

    const sources = extractSources(
      response.candidates?.[0]?.groundingMetadata?.groundingChunks as RawGroundingChunk[] | undefined
    );

    const plantObj = parsed.plant as {
      scientificName?: string;
      nameEn?: string;
      nameKo?: string;
    };
    const imageUrl = await resolvePlantPhotoUrl({
      scientificName: plantObj.scientificName,
      nameEn: plantObj.nameEn,
      nameKo: plantObj.nameKo,
    });

    res.json({
      plant: {
        ...parsed.plant,
        ...(imageUrl ? { imageUrl } : {}),
        sources,
        isGoogleLive,
      },
      sources,
    });
  } catch (error) {
    console.error('[/api/plants/search] error:', error);
    const { status, message, quotaExceeded } = mapApiError(error);
    res.status(status).json({ error: message, quotaExceeded });
  }
});

app.post('/api/plants/recommend', async (req, res) => {
  try {
    const description =
      typeof req.body?.description === 'string' ? req.body.description.trim() : '';
    const similarToPlant =
      typeof req.body?.similarToPlant === 'string'
        ? req.body.similarToPlant.trim().slice(0, 80)
        : '';
    const traits = req.body?.traits as
      | {
          appearance?: string;
          color?: string;
          scent?: string;
          environment?: string;
          season?: string;
        }
      | undefined;

    const combinedParts = [
      similarToPlant
        ? `기준 식물: ${similarToPlant} (이 식물 자체는 제외하고, 생김새·색감·향기·키우는 환경이 비슷한 다른 식물 추천)`
        : description,
      similarToPlant && description && description !== similarToPlant
        ? `원본 입력: ${description}`
        : '',
      traits?.season ? `현재 계절(우선 추천): ${traits.season}` : '',
      traits?.appearance ? `생김새: ${traits.appearance}` : '',
      traits?.color ? `색상: ${traits.color}` : '',
      traits?.scent ? `향기/냄새: ${traits.scent}` : '',
      traits?.environment ? `키우는 환경: ${traits.environment}` : '',
    ].filter(Boolean);

    const rawQuery = combinedParts.join(' / ');
    if (!rawQuery) {
      res.status(400).json({ error: '원하는 생김새, 색상, 향기 등을 입력해 주세요.' });
      return;
    }
    const finalQuery = rawQuery.slice(0, 340);

    const ai = getGeminiClient();

    const similarityRule = similarToPlant
      ? `중요 규칙: 기준 식물인 "${similarToPlant}" 자체는 절대 recommendations에 포함하지 마세요. 대신 "${similarToPlant}"와 생김새·잎/꽃 색감·향기·분류학적 근연 관계·생육 환경이 가장 비슷한 **다른 실존 식물 4~6종**을 선정하고, matchReason에 "${similarToPlant}"와 어떤 점이 비슷한지 1~2문장으로 구체적으로 설명하세요.`
      : `중요 규칙: 만약 사용자가 특정 식물 이름(예: '몬스테라', '라벤더', '장미', '벚꽃', '해피트리' 등)이나 '~와 비슷한 식물'을 입력했다면, 그 식물 자체는 recommendations에 포함하지 말고, 그 식물과 생김새·색감·향기·분류·키우는 환경이 가장 비슷한 **다른 실존 식물 4~6종**을 선정해 어떤 점이 닮았는지 matchReason에 설명하세요.`;

    const prompt = `사용자가 원하는 식물 조건: "${finalQuery}"

${similarityRule}
위 조건(생김새, 색감, 냄새·향기, 환경 등)에 가장 잘 어울리는 실제 식물 4~6종을 선정해 주세요.
한국에서 실제로 가정이나 정원에서 키울 수 있는 정확한 실존 식물만 추천하세요.

반드시 마크다운 설명 없이 순수 JSON 객체 하나만 출력하세요. 형식은 다음과 같습니다:
{
  "querySummary": "${similarToPlant ? `‘${similarToPlant}’와 비슷한 식물` : '사용자 조건을 한 줄로 정리한 문장'}",
  "recommendations": [
    {
      "nameKo": "한국어 대표 식물 이름",
      "nameEn": "영문 이름",
      "scientificName": "라틴어 학명",
      "category": "다음 중 정확히 하나 선택: 관엽식물 | 허브·방향식물 | 다육·선인장 | 꽃·관화식물 | 공기정화·야자",
      "difficulty": "다음 중 정확히 하나 선택: 초보자 추천 | 보통 | 세심한 관리 필요",
      "matchReason": "사용자가 요청한 기준 식물 또는 생김새·색·향기 조건과 왜 비슷하고 잘 맞는지 설명 (2문장 이내)",
      "appearanceSummary": "잎·꽃·수형 생김새 요약",
      "colorDescription": "주된 색상 묘사",
      "colorHexes": ["#HEX1", "#HEX2"],
      "scentSummary": "향기/냄새 특징",
      "quickCareTip": "핵심 관리 요령 한 줄"
    }
  ]
}`;

    const { response, isGoogleLive } = await generateWithFallback(
      ai,
      prompt,
      '당신은 감각적인 식물 큐레이터이자 원예 전문가입니다. 사용자가 생김새·색감·향기를 설명하면 어울리는 식물을, 특정 식물 이름을 입력하면 그 식물 자체는 제외하고 생김새·색감·향기·특성이 가장 비슷한 다른 실존 식물을 정확히 매칭해 추천합니다. 출력은 오직 유효한 JSON 객체 하나만 반환합니다.'
    );

    const parsed = extractJsonFromText<{
      querySummary?: string;
      recommendations?: unknown[];
    }>(response.text ?? '');

    const sources = extractSources(
      response.candidates?.[0]?.groundingMetadata?.groundingChunks as RawGroundingChunk[] | undefined
    );

    const normalizeForFilter = (v: string) =>
      v.toLowerCase().replace(/\s*\([^)]*\)\s*/g, '').replace(/\s+/g, '');
    const normalizedSimilarTarget = similarToPlant ? normalizeForFilter(similarToPlant) : '';

    const rawRecommendations = (
      Array.isArray(parsed.recommendations)
        ? (parsed.recommendations as Array<Record<string, unknown>>)
        : []
    ).filter((rec) => {
      if (!normalizedSimilarTarget) return true;
      const nameKo = typeof rec.nameKo === 'string' ? normalizeForFilter(rec.nameKo) : '';
      return nameKo !== normalizedSimilarTarget;
    });

    const resolvedImages = await resolvePlantPhotosBatch(
      rawRecommendations.map((rec) => ({
        scientificName:
          typeof rec.scientificName === 'string' ? rec.scientificName : undefined,
        nameEn: typeof rec.nameEn === 'string' ? rec.nameEn : undefined,
        nameKo: typeof rec.nameKo === 'string' ? rec.nameKo : undefined,
      }))
    );

    const recommendations = rawRecommendations.map((rec, idx) => ({
      ...rec,
      ...(resolvedImages[idx] ? { imageUrl: resolvedImages[idx] } : {}),
    }));

    res.json({
      querySummary:
        parsed.querySummary ||
        (similarToPlant ? `‘${similarToPlant}’와 비슷한 식물` : finalQuery),
      recommendations,
      sources,
      sourceMode: isGoogleLive ? 'google-live' : 'gemini-knowledge',
    });
  } catch (error) {
    console.error('[/api/plants/recommend] error:', error);
    const { status, message, quotaExceeded } = mapApiError(error);
    res.status(status).json({ error: message, quotaExceeded });
  }
});

interface SearchStatEntry {
  term: string;
  count: number;
  lastSearchedAt: number;
}

const MAX_TRACKED_SEARCHES = 500;
const searchStats = new Map<string, SearchStatEntry>();
const searchSseClients = new Set<Response>();

function normalizeTrackedSearchTerm(raw: unknown): string | null {
  if (typeof raw !== 'string') return null;
  const cleaned = raw
    .replace(/<[^>]*>/g, '')
    .replace(/\s*\([^)]*\)\s*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned || cleaned.length > 35) return null;
  if (/^[ㄱ-ㅎㅏ-ㅣ]$/.test(cleaned)) return null;
  return cleaned;
}

function getPopularSearchPayload() {
  const terms = Array.from(searchStats.values())
    .sort((a, b) => b.count - a.count || b.lastSearchedAt - a.lastSearchedAt)
    .slice(0, 12);
  return {
    terms,
    queries: terms.slice(0, 6).map((t) => t.term),
  };
}

function broadcastPopularSearches() {
  if (searchSseClients.size === 0) return;
  const data = `data: ${JSON.stringify(getPopularSearchPayload())}\n\n`;
  for (const client of searchSseClients) {
    try {
      client.write(data);
    } catch {
      searchSseClients.delete(client);
    }
  }
}

function recordSearchStat(rawQuery: unknown) {
  const term = normalizeTrackedSearchTerm(rawQuery);
  if (!term) return null;

  const key = term.toLowerCase();
  const existing = searchStats.get(key);
  const now = Date.now();
  if (existing) {
    existing.count += 1;
    existing.lastSearchedAt = now;
  } else {
    if (searchStats.size >= MAX_TRACKED_SEARCHES) {
      let oldestKey: string | null = null;
      let oldestTime = Infinity;
      for (const [k, v] of searchStats.entries()) {
        if (v.count <= 1 && v.lastSearchedAt < oldestTime) {
          oldestTime = v.lastSearchedAt;
          oldestKey = k;
        }
      }
      if (!oldestKey) {
        oldestKey = searchStats.keys().next().value ?? null;
      }
      if (oldestKey) searchStats.delete(oldestKey);
    }
    searchStats.set(key, { term, count: 1, lastSearchedAt: now });
  }

  broadcastPopularSearches();
  return getPopularSearchPayload();
}

app.get('/api/searches/popular', (_req, res) => {
  res.json(getPopularSearchPayload());
});

app.post('/api/searches/record', (req, res) => {
  const payload = recordSearchStat(req.body?.query);
  if (!payload) {
    res.status(400).json({ error: '유효한 검색어가 아닙니다.' });
    return;
  }
  res.json(payload);
});

app.get('/api/searches/stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  res.write(`data: ${JSON.stringify(getPopularSearchPayload())}\n\n`);
  searchSseClients.add(res);

  const heartbeat = setInterval(() => {
    try {
      res.write(': ping\n\n');
    } catch {
      clearInterval(heartbeat);
      searchSseClients.delete(res);
    }
  }, 25000);

  req.on('close', () => {
    clearInterval(heartbeat);
    searchSseClients.delete(res);
  });
});

app.get('/api/project/download-zip', (_req, res) => {
  const zipPath = path.resolve(__dirname, 'allplant.zip');
  if (!fs.existsSync(zipPath)) {
    res.status(404).json({ error: 'ZIP 파일을 찾을 수 없습니다.' });
    return;
  }
  res.setHeader('Content-Type', 'application/zip');
  res.download(zipPath, 'allplant.zip');
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
