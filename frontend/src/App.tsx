import { useState, useEffect } from "react";

function App() {
  const [messages, setMessages] = useState<Array<{ text: string; isUser: boolean }>>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [appReady, setAppReady] = useState(false);

  // Poll backend status on mount
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    const checkStatus = async () => {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        if (data.ready) {
          setAppReady(true);
          if (interval) clearInterval(interval);
        }
      } catch (e) {
        // ignore errors, keep polling
      }
    };
    checkStatus();
    interval = setInterval(checkStatus, 1500);
    return () => clearInterval(interval);
  }, []);

  async function sendMessage() {
    if (!inputText.trim()) return;

    const userMessage = { text: inputText, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setInputText("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputText }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      const fullText = data.response + (data.page_references ? `\n\nReferences:\n${data.page_references.join("\n")}` : "");

      setMessages(prev => [...prev, { text: fullText, isUser: false }]);
      speakText(fullText);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prev => [...prev, { text: "Sorry, there was an error processing your request.", isUser: false }]);
    } finally {
      setIsLoading(false);
    }
  }

  async function sendMessageVoice(text: string) {
    if (!text.trim()) return;

    const userMessage = { text, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      var fullText = "";
      
      const data = await response.json();
      
      const assistantMessage = {
  text: data.response,
  isUser: false,
  pages: data.page_references || []
};
setMessages(prev => [...prev, assistantMessage]);
speakText(data.response);

       
      setMessages(prev => [...prev, { text: fullText, isUser: false }]);
      speakText(data.response);

    } catch (error) {
      console.error("Error sending voice message:", error);
      setMessages(prev => [...prev, { text: "Sorry, there was an error processing your voice request.", isUser: false }]);
    } finally {
      setIsLoading(false);
    }
  }

  function startListening() {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech Recognition API not supported in this browser.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "el-GR";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInputText(transcript);
      sendMessageVoice(transcript);
    };

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error", event.error);
    };

    recognition.start();
  }

  function speakText(text: string) {
  if (!window.speechSynthesis) return;

  const utterance = new SpeechSynthesisUtterance(text);
  // Try to find a Greek voice
  const voices = window.speechSynthesis.getVoices();
  const greekVoice = voices.find(voice => voice.lang.startsWith("el"));

  if (greekVoice) {
    utterance.voice = greekVoice;
    utterance.lang = greekVoice.lang; // set lang to match voice
  } else {
    // fallback to default Greek locale
    utterance.lang = "el-GR";
  }

  window.speechSynthesis.speak(utterance);
}

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!appReady) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100">
        <div className="flex flex-col items-center">
          <div className="flex space-x-2 mb-4">
            <div className="w-4 h-4 bg-blue-600 rounded-full animate-bounce"></div>
            <div className="w-4 h-4 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
            <div className="w-4 h-4 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></div>
          </div>
          <div className="text-blue-700 font-semibold text-lg">Loading application...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <div className="bg-white rounded-xl shadow-lg flex-1 flex flex-col max-w-4xl mx-auto w-full overflow-hidden">
        <div className="p-6 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold text-xl flex items-center">
          <svg className="w-8 h-8 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          BMW X1 Manual Assistant
        </div>

        <div className="flex-1 p-6 overflow-y-auto bg-gray-50">
          {messages.length === 0 && (
            <div className="text-gray-500 text-center mt-8 space-y-4">
              <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-lg">Ask any question about the BMW X1 manual</p>
            </div>
          )}

          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`mb-6 flex ${message.isUser ? "justify-end" : "justify-start"}`}
            >
              <div 
                className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${
                  message.isUser 
                    ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-none" 
                    : "bg-white text-gray-800 rounded-bl-none border border-gray-100"
                }`}
              >
                {message.text}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start mb-6">
              <div className="bg-white text-gray-800 p-4 rounded-2xl rounded-bl-none shadow-sm border border-gray-100">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-gray-100 bg-white">
          <div className="flex gap-2">
            <textarea
              className="flex-1 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Type your message..."
              rows={1}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
            />
            <button
              className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-3 rounded-xl hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50"
              onClick={sendMessage}
              disabled={isLoading || !inputText.trim()}
              aria-label="Send message"
            >
              ➤
            </button>
            <button
              className="bg-gradient-to-r from-green-600 to-green-700 text-white px-4 py-3 rounded-xl hover:from-green-700 hover:to-green-800 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50"
              onClick={startListening}
              disabled={isLoading}
              aria-label="Start voice input"
            >
              🎤
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
