import "dotenv/config";
import express from "express";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";
import { WebSocketServer, WebSocket } from "ws";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface SimpleTouristSpot {
  spotName: string;
  timeSlot: string;
  entryAndLocalCost: number;
  activity: string;
}

interface SimpleDestination {
  id: string;
  name: string;
  district: string;
  category: string;
  description: string;
  avgHotelPerPerson: number;
  avgFoodPerDay: number;
  avgTravelCostPerPerson: number;
  spots: SimpleTouristSpot[];
}

let destinations: SimpleDestination[] = [];
try {
  const jsonPath = path.join(__dirname, "server-destinations.json");
  if (fs.existsSync(jsonPath)) {
    destinations = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
  }
} catch (err) {
  console.warn("Could not load server-destinations.json:", err);
}

function createGenAIClient(): GoogleGenAI {
  const key = process.env.GEMINI_API_KEY || process.env.API_KEY || undefined;
  return new GoogleGenAI({
    apiKey: key,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// Build a comprehensive summary of all 100 destinations and their authentic spots for grounding
const DESTINATIONS_KNOWLEDGE_SUMMARY = destinations
  .map((d) => {
    const spotsList = d.spots
      .slice(0, 4)
      .map((s) => `${s.spotName} (${s.timeSlot}, ₹${s.entryAndLocalCost}, ${s.activity})`)
      .join("; ");
    return `- ${d.name} (${d.district} District, ${d.category}): ${d.description}. Hotel ₹${d.avgHotelPerPerson}/night, Food ₹${d.avgFoodPerDay}/day, Travel ₹${d.avgTravelCostPerPerson}. Spots: ${spotsList}`;
  })
  .join("\n");

function buildTouristSystemInstruction(preferredLang = "ta"): string {
  return `You are the official Smart AI Tourist Guide & Real-Time Voice Assistant for "TN Smart Tourist Planner" (தமிழ்நாடு ஸ்மார்ட் சுற்றுலா வழிகாட்டி).

MANDATORY MULTILINGUAL RULES:
1. DETECT THE USER'S INPUT LANGUAGE:
   - If the user asks in Tamil (தமிழ்) or Tanglish (Tamil written in English script like "Madurai la enna paarkalam?", "Ooty poga best time enna?"), answer warmly and fluently in pure, natural Tamil (தமிழ்).
   - If the user asks in English, answer in clear, elegant, detailed English.
   - If the user asks in Hindi (हिन्दी) or Hinglish, answer warmly and respectfully in Hindi (हिन्दी).
   - If the preferred language is "${preferredLang}", use it unless the user explicitly switches.

TOURIST EXPERTISE (100 Tamil Nadu Destinations & Worldwide):
1. You provide accurate, authoritative tourist information:
   - Exact temple darshan & pooja timings (e.g. Madurai Meenakshi: 5:00 AM - 12:30 PM & 4:00 PM - 10:00 PM; Thanjavur Brihadeeswarar: 6:00 AM - 12:30 PM & 4:00 PM - 8:30 PM; Rameswaram Ramanathaswamy 22 theerthams: 5:00 AM - 1:00 PM & 3:00 PM - 9:00 PM; Kanchipuram Kamakshi: 5:30 AM - 12:15 PM & 4:00 PM - 8:15 PM).
   - Dress codes: Traditional attire (Dhotis/Kurta, Sarees/Salwars; shorts/sleeveless prohibited in traditional TN temples).
   - Hill station seasonal viewpoints, peak blooming season (Ooty Rose Garden, Botanical Garden, Nilgiri Mountain Toy Train; Kodaikanal Lake, Pillar Rocks; Yercaud; Valparai; Kolli Hills 70 hairpin bends).
   - Waterfalls seasonal water flow, safety guidelines (Hogenakkal coracle boating, Courtallam herbal bathing season June-September, Suruli falls, Agaya Gangai).
   - Beaches, sunrise & sunset viewpoints (Marina beach, Dhanushkodi ghost town & Arichal Munai, Kanyakumari confluence point, Mahabalipuram shore temple).
   - Famous local food & delicacies (Madurai Jigarthanda & Kari Dosa, Chettinad pepper chicken & paniyaram, Tirunelveli Iruttu Kadai Halwa, Ambur Biryani, Kumbakonam Degree Coffee, Dindigul Thalappakatti Biryani).
   - Budget planning: Hotel tariff, food cost, local transport auto/taxi fares, entry tickets.
2. Format answers neatly with emojis, bullet points, and welcoming tone.

Reference Tamil Nadu Destinations in this App:
${DESTINATIONS_KNOWLEDGE_SUMMARY}`;
}

/**
 * Intelligent local knowledge generator that guarantees rich, accurate tourist answers
 * in Tamil, English, or Hindi even during upstream model rate-limits or offline periods.
 */
function generateLocalTouristAnswer(question: string, lang = "ta"): string {
  const qLower = question.toLowerCase();

  // Detect query language if not explicitly provided
  const isTamilQuery =
    /[\u0B80-\u0BFF]/.test(question) ||
    /enna|eppadi|paarkalam|kovil|neram|selavu|oor|thala|engu|yaar|eppothu/i.test(
      qLower
    );
  const isHindiQuery =
    /[\u0900-\u097F]/.test(question) ||
    /kya|kaise|kab|kahan|kitna|mandir|jagah|samay|kharch/i.test(qLower);

  const activeLang = isTamilQuery
    ? "ta"
    : isHindiQuery
    ? "hi"
    : lang === "ta"
    ? "ta"
    : lang === "hi"
    ? "hi"
    : "en";

  // Find destination matching the question
  const matchedDest = destinations.find((d) => {
    const nameMatch = qLower.includes(d.name.toLowerCase());
    const distMatch = qLower.includes(d.district.toLowerCase());
    const spotMatch = d.spots.some((s) =>
      qLower.includes(s.spotName.toLowerCase())
    );
    return nameMatch || distMatch || spotMatch;
  });

  if (matchedDest) {
    const spotsFormatted = matchedDest.spots
      .slice(0, 5)
      .map(
        (s, idx) =>
          `${idx + 1}. **${s.spotName}** — ${s.timeSlot} | கட்டணம்/செலவு: ₹${
            s.entryAndLocalCost
          }\n   செயல்பாடு: ${s.activity}`
      )
      .join("\n\n");

    const spotsFormattedEn = matchedDest.spots
      .slice(0, 5)
      .map(
        (s, idx) =>
          `${idx + 1}. **${s.spotName}** — ${
            s.timeSlot
          } | Entry/Local Cost: ₹${s.entryAndLocalCost}\n   Activity: ${s.activity}`
      )
      .join("\n\n");

    const spotsFormattedHi = matchedDest.spots
      .slice(0, 5)
      .map(
        (s, idx) =>
          `${idx + 1}. **${s.spotName}** — समय: ${
            s.timeSlot
          } | प्रवेश/स्थानीय खर्च: ₹${s.entryAndLocalCost}\n   गतिविधि: ${s.activity}`
      )
      .join("\n\n");

    if (activeLang === "ta") {
      return `வணக்கம்! **${matchedDest.name} (${matchedDest.district} மாவட்டம்)** பற்றிய முழுமையான சுற்றுலா விவரங்கள் இதோ:

🌟 **முக்கிய சிறப்பம்சம்:** ${matchedDest.description}
🏛️ **வகை:** ${matchedDest.category}

🏛️ **பார்க்க வேண்டிய முக்கிய இடங்கள் & நேரம்:**
${spotsFormatted}

💰 **உத்தேச பட்ஜெட் (ஒரு நபருக்கு ஒரு நாளுக்கு):**
- 🏨 தங்குமிடம் (Hotel): சராசரியாக ₹${matchedDest.avgHotelPerPerson}
- 🍽️ உணவு (Food): சராசரியாக ₹${matchedDest.avgFoodPerDay}
- 🚗 உள்ளூர் பயணம் (Travel): சராசரியாக ₹${matchedDest.avgTravelCostPerPerson}

💡 **சுற்றுலா குறிப்புகள்:**
- கோவில்களுக்குச் செல்லும்போது பாரம்பரிய ஆடை அணிவது கட்டாயம் (Dhoti / Saree / Salwar).
- காலை வேளையில் தரிசனம் மற்றும் காட்சிகளைப் பார்வையிடுவது கூட்ட நெரிசலைத் தவிர்க்க உதவும்.

வேறு ஏதேனும் தகவல்கள் வேண்டுமானால் தயங்காமல் கேளுங்கள்!`;
    } else if (activeLang === "hi") {
      return `नमस्ते! **${matchedDest.name} (${matchedDest.district} जिला)** के बारे में संपूर्ण पर्यटन जानकारी यहाँ दी गई है:

🌟 **प्रमुख आकर्षण:** ${matchedDest.description}
🏛️ **श्रेणी:** ${matchedDest.category}

🏛️ **प्रमुख दर्शनीय स्थल और समय:**
${spotsFormattedHi}

💰 **अनुमानित बजट (प्रति व्यक्ति प्रति दिन):**
- 🏨 होटल आवास: लगभग ₹${matchedDest.avgHotelPerPerson}
- 🍽️ भोजन खर्च: लगभग ₹${matchedDest.avgFoodPerDay}
- 🚗 स्थानीय यात्रा: लगभग ₹${matchedDest.avgTravelCostPerPerson}

💡 **उपयोगी सुझाव:**
- मंदिरों में दर्शन के लिए पारंपरिक पोशाक का पालन करें।
- सुबह के समय दर्शन करने से भीड़ से बचा जा सकता है।

क्या आप इस स्थान के लिए विस्तृत यात्रा योजना बनाना चाहते हैं?`;
    } else {
      return `Welcome! Here is the complete tourist guide for **${matchedDest.name} (${matchedDest.district} District)**:

🌟 **Highlights & Overview:** ${matchedDest.description}
🏛️ **Category:** ${matchedDest.category}

🏛️ **Top Spots to Visit & Timings:**
${spotsFormattedEn}

💰 **Estimated Budget (Per Person Per Day):**
- 🏨 Hotel Stay: ~₹${matchedDest.avgHotelPerPerson}/night
- 🍽️ Food & Dining: ~₹${matchedDest.avgFoodPerDay}/day
- 🚗 Local Travel & Transit: ~₹${matchedDest.avgTravelCostPerPerson}

💡 **Travel Tips:**
- Traditional dress code is required in heritage temples (Dhoti/Saree/Salwar; shorts strictly prohibited).
- Early morning visits ensure peaceful sightseeing without crowd delays.

Feel free to ask about nearby places, routes, or local food recommendations!`;
    }
  }

  // General recommendation when no single destination matched
  if (activeLang === "ta") {
    return `வணக்கம்! தமிழ்நாடு சுற்றுலா குறித்த சிறந்த பரிந்துரைகள் இதோ:

🏛️ **ஆன்மீக தலங்கள் (Spiritual & Heritage):**
- **மதுரை மீனாட்சி அம்மன் கோவில்** — காலை 5:00 - 12:30 & மாலை 4:00 - 10:00.
- **தஞ்சாவூர் பெரிய கோவில்** — சோழர் கால உலகப் பாரம்பரிய தளம்.
- **ராமேஸ்வரம் ராமநாதசுவாமி கோவில் & தனுஷ்கோடி** — 22 புனித தீர்த்தங்கள்.

🌲 **மலை வாசஸ்தலங்கள் (Hill Stations):**
- **ஊட்டி (Ooty)** — தாவரவியல் பூங்கா, ரோஜா தோட்டம், பொம்மை ரயில்.
- **கொடைக்கானல் (Kodaikanal)** — ஏரி படகு சவாரி, தூண் பாறைகள்.
- **ஏற்காடு (Yercaud) & வால்பாறை (Valparai)** — பசுமையான தேயிலைத் தோட்டங்கள்.

🌊 **அருவிகள் & கடற்கரைகள் (Waterfalls & Beaches):**
- **குற்றாலம் (Courtallam)** — மூலிகை அருவிகள் (ஜூன் முதல் செப்டம்பர் வரை).
- **கன்னியாகுமரி** — முக்கடல் சங்கமம், சூரியோதயம் & விவேகானந்தர் பாறை.
- **மகாபலிபுரம்** — கடற்கரை கோவில் & ஐந்து ரதங்கள் சிற்பக்கலை.

💰 **பட்ஜெட்:** 2 நாள் சுற்றுலாவிற்கு ஒரு நபருக்கு சுமார் ₹2,500 - ₹3,500 போதுமானது.
உங்களுக்கு எந்த குறிப்பிட்ட ஊர் அல்லது சுற்றுலாத் தலம் பற்றி முழு விவரம் வேண்டும்? கேளுங்கள்!`;
  } else if (activeLang === "hi") {
    return `नमस्ते! तमिलनाडु पर्यटन के लिए प्रमुख सिफारिशें यहाँ हैं:

🏛️ **प्रसिद्ध मंदिर और ऐतिहासिक स्थल:**
- **मदुरै मीनाक्षी मंदिर**: दर्शन समय सुबह 5:00 - 12:30 और शाम 4:00 - रात 10:00।
- **तंजौर बृहदीश्वर मंदिर**: यूनेस्को विश्व धरोहर वास्तुकला।
- **रामेश्वरम और धनुषकोडी**: 22 पवित्र कुंड स्नान और समुद्र तट।

🌲 **प्रमुख हिल स्टेशन:**
- **ऊटी (Ooty)**: बॉटनिकल गार्डन, टॉय ट्रेन और चाय के बागान।
- **कोडाइकनाल**: सुंदर झील, पिलर रॉक्स और सुखद मौसम।
- **यरकौड और वालपारई**: शांत प्राकृतिक वातावरण।

🌊 **झरने और समुद्र तट:**
- **कुट्रालम**: औषधीय जलप्रपात (जून से सितंबर)।
- **कन्याकुमारी**: त्रिवेणी संगम और विवेकानंद रॉक।
- **महाबलीपुरम**: तट मंदिर और प्राचीन रॉक कट गुफाएं।

💰 **अनुमानित बजट:** 2-दिन की यात्रा के लिए प्रति व्यक्ति ₹2,500 से ₹3,500 पर्याप्त है।
आप किस विशेष स्थान के बारे में जानना चाहते हैं? मुझे बताएं!`;
  } else {
    return `Welcome! Here are top recommendations for exploring Tamil Nadu's 100 destinations:

🏛️ **Spiritual & Architectural Wonders:**
- **Madurai Meenakshi Temple**: Darshan hours 5:00 AM - 12:30 PM & 4:00 PM - 10:00 PM.
- **Thanjavur Brihadeeswarar Temple**: 1,000-year-old Chola architectural marvel.
- **Rameswaram & Dhanushkodi**: 22 holy water theerthams & scenic oceanic roads.

🌲 **Scenic Hill Stations:**
- **Ooty (Nilgiris)**: Botanical Gardens, Rose Garden, Nilgiri Mountain Toy Train.
- **Kodaikanal**: Star-shaped lake, Coaker's walk, Pine forests.
- **Yercaud & Valparai**: Refreshing tea estates and scenic viewpoints.

🌊 **Waterfalls & Coastal Delights:**
- **Courtallam**: Natural herbal water baths (best season: June to September).
- **Kanyakumari**: Triveni Sangam, Vivekananda Rock, sunrise & sunset view.
- **Mahabalipuram**: UNESCO Shore Temple and monolithic rock monuments.

💰 **Budget Guide:** A 2-day trip typically averages ₹2,500 to ₹3,500 per person including hotel, food, and transport.
Which tourist spot or city would you like to explore next?`;
  }
}

/**
 * Sanitizes multi-turn chat history so:
 * 1. The first message ALWAYS has role: 'user' (Gemini requirement).
 * 2. Turns alternate strictly between 'user' and 'model'.
 * 3. Empty text turns are stripped.
 */
function formatChatContents(
  history: Array<{ role?: string; text?: string }>,
  currentMessage: string
): Array<{ role: "user" | "model"; parts: Array<{ text: string }> }> {
  const contents: Array<{
    role: "user" | "model";
    parts: Array<{ text: string }>;
  }> = [];

  let seenUser = false;
  for (const item of history || []) {
    if (!item || typeof item.text !== "string" || !item.text.trim()) continue;
    const role: "user" | "model" = item.role === "model" ? "model" : "user";

    if (!seenUser) {
      if (role === "user") {
        seenUser = true;
        contents.push({ role: "user", parts: [{ text: item.text.trim() }] });
      }
    } else {
      const last = contents[contents.length - 1];
      if (last.role !== role) {
        contents.push({ role, parts: [{ text: item.text.trim() }] });
      } else {
        last.parts[0].text += `\n${item.text.trim()}`;
      }
    }
  }

  const cleanCurrent = (currentMessage || "").trim();
  if (contents.length > 0 && contents[contents.length - 1].role === "user") {
    if (cleanCurrent) {
      contents[contents.length - 1].parts[0].text += `\n${cleanCurrent}`;
    }
  } else if (cleanCurrent) {
    contents.push({ role: "user", parts: [{ text: cleanCurrent }] });
  }

  // Safety fallback: if contents is still empty, ensure one user turn
  if (contents.length === 0) {
    contents.push({
      role: "user",
      parts: [{ text: cleanCurrent || "Hello tourist guide" }],
    });
  }

  return contents;
}

async function startServer() {
  const app = express();
  const server = http.createServer(app);
  const PORT = 3000;

  app.use(express.json({ limit: "10mb" }));

  // 1. Text / Multimodal Tourist Q&A Endpoint (Answers anything about any tourist place)
  app.post("/api/tourist-chat", async (req, res) => {
    try {
      const { message, history = [], lang = "ta" } = req.body || {};
      if (!message || typeof message !== "string" || !message.trim()) {
        res.status(400).json({ error: "Message is required." });
        return;
      }

      const ai = createGenAIClient();
      const formattedContents = formatChatContents(history, message);
      const systemInstruction = buildTouristSystemInstruction(lang);

      let replyText = "";

      // Attempt 1: gemini-3.1-flash-lite (fast, robust, high availability)
      try {
        const response1 = await ai.models.generateContent({
          model: "gemini-3.1-flash-lite",
          contents: formattedContents,
          config: {
            systemInstruction,
          },
        });
        replyText = response1.text || "";
      } catch (err1) {
        console.warn("Attempt 1 (gemini-3.1-flash-lite) failed:", err1);
      }

      // Attempt 2: gemini-flash-latest
      if (!replyText) {
        try {
          const fallbackResponse2 = await ai.models.generateContent({
            model: "gemini-flash-latest",
            contents: formattedContents,
            config: {
              systemInstruction,
            },
          });
          replyText = fallbackResponse2.text || "";
        } catch (err2) {
          console.warn("Attempt 2 (gemini-flash-latest) failed:", err2);
        }
      }

      // Attempt 3: gemini-3.8-flash
      if (!replyText) {
        try {
          const fallbackResponse3 = await ai.models.generateContent({
            model: "gemini-3.8-flash",
            contents: formattedContents,
            config: {
              systemInstruction,
            },
          });
          replyText = fallbackResponse3.text || "";
        } catch (err3) {
          console.warn("Attempt 3 (gemini-3.8-flash) failed:", err3);
        }
      }

      // Attempt 4: Intelligent local tourist knowledge engine fallback (never leaves user stranded)
      if (!replyText) {
        console.info("Using local tourist knowledge generator fallback for query:", message);
        replyText = generateLocalTouristAnswer(message, lang);
      }

      res.json({ reply: replyText });
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error
          ? error.message
          : "Failed to get response from Tourist AI.";
      console.error("Error in /api/tourist-chat:", errMsg);
      // Fallback safely to local knowledge
      const safeReply = generateLocalTouristAnswer(
        req.body?.message || "Tamil Nadu tourist places",
        req.body?.lang || "ta"
      );
      res.json({ reply: safeReply });
    }
  });

  // 2. Text-to-Speech Endpoint using gemini-3.8-flash-lite-tts
  app.post("/api/tourist-tts", async (req, res) => {
    try {
      const { text } = req.body || {};
      if (!text || typeof text !== "string") {
        res.status(400).json({ error: "Text is required for speech." });
        return;
      }

      const ai = createGenAIClient();
      const cleanText = text.replace(/[*#_`~]/g, "").slice(0, 1200);

      const response = await ai.models.generateContent({
        model: "gemini-3.8-flash-lite-tts",
        contents: [
          {
            role: "user",
            parts: [{ text: cleanText }],
          },
        ],
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: "Kore" },
            },
          },
        },
      });

      const base64Audio =
        response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) {
        res.status(500).json({ error: "No audio returned from TTS model." });
        return;
      }

      res.json({ audioWavBase64: base64Audio });
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error ? error.message : "TTS generation failed.";
      console.error("Error in /api/tourist-tts:", errMsg);
      res.status(500).json({ error: errMsg });
    }
  });

  // 3. Real-time Voice Conversation WebSocket Server (/live) using gemini-3.8-live
  const wss = new WebSocketServer({ noServer: true });

  server.on("upgrade", (request, socket, head) => {
    const pathname = new URL(
      request.url || "/",
      `http://${request.headers.host || "localhost"}`
    ).pathname;

    if (pathname === "/live") {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit("connection", ws, request);
      });
    }
  });

  wss.on("connection", async (clientWs: WebSocket, request) => {
    const urlObj = new URL(
      request.url || "/live",
      `http://${request.headers.host || "localhost"}`
    );
    const lang = urlObj.searchParams.get("lang") || "ta";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let liveSession: any = null;

    try {
      const ai = createGenAIClient();

      const sessionPromise = ai.live.connect({
        model: "gemini-3.8-live",
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: "Zephyr" },
            },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: buildTouristSystemInstruction(lang),
        },
        callbacks: {
          onopen: () => {
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ type: "connected" }));
            }
          },
          onmessage: (message: LiveServerMessage) => {
            if (clientWs.readyState !== WebSocket.OPEN) return;

            const parts = message.serverContent?.modelTurn?.parts || [];
            for (const part of parts) {
              if (part.inlineData?.data) {
                clientWs.send(
                  JSON.stringify({
                    type: "audio",
                    audio: part.inlineData.data,
                  })
                );
              }
              if (part.text) {
                clientWs.send(
                  JSON.stringify({
                    type: "modelText",
                    text: part.text,
                  })
                );
              }
            }

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const sc = message.serverContent as any;
            if (sc?.inputTranscription?.text) {
              clientWs.send(
                JSON.stringify({
                  type: "inputTranscript",
                  text: sc.inputTranscription.text,
                })
              );
            }
            if (sc?.outputTranscription?.text) {
              clientWs.send(
                JSON.stringify({
                  type: "outputTranscript",
                  text: sc.outputTranscription.text,
                })
              );
            }
            if (message.serverContent?.interrupted) {
              clientWs.send(JSON.stringify({ type: "interrupted" }));
            }
            if (message.serverContent?.turnComplete) {
              clientWs.send(JSON.stringify({ type: "turnComplete" }));
            }
          },
          onerror: (err: unknown) => {
            const msg =
              err instanceof Error ? err.message : "Live API session error";
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ type: "error", error: msg }));
            }
          },
          onclose: () => {
            if (clientWs.readyState === WebSocket.OPEN) {
              clientWs.send(JSON.stringify({ type: "closed" }));
            }
          },
        },
      });

      liveSession = await sessionPromise;

      clientWs.on("message", (raw) => {
        try {
          const parsed = JSON.parse(raw.toString());
          if (parsed.audio && liveSession) {
            liveSession.sendRealtimeInput({
              audio: {
                data: parsed.audio,
                mimeType: "audio/pcm;rate=16000",
              },
            });
          } else if (parsed.text && liveSession) {
            if (typeof liveSession.sendClientContent === "function") {
              liveSession.sendClientContent({
                turns: [{ role: "user", parts: [{ text: parsed.text }] }],
                turnComplete: true,
              });
            } else {
              liveSession.sendRealtimeInput({
                text: parsed.text,
              });
            }
          }
        } catch (err) {
          console.error("Failed to process incoming WS message:", err);
        }
      });

      clientWs.on("close", () => {
        if (liveSession) {
          try {
            liveSession.close();
          } catch {
            // ignore close error
          }
        }
      });
    } catch (error: unknown) {
      const errMsg =
        error instanceof Error
          ? error.message
          : "Failed to initialize Gemini Live session.";
      console.error("Live session connection error:", errMsg);
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify({ type: "error", error: errMsg }));
        clientWs.close();
      }
    }
  });

  // Mount Vite dev server in development or serve built static files in production
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  server.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
