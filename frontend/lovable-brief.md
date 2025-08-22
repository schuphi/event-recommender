# 🎵 Copenhagen Event Recommender - Lovable Brief

## Project Overview
Create a modern, dark-themed React frontend for a Copenhagen nightlife event discovery app. The app uses machine learning to recommend events based on user preferences and provides an engaging, swipe-friendly interface for event discovery.

## Core Features to Implement

### 1. 📱 Main Event Feed
- **Infinite scroll** event recommendation feed
- **Card-based layout** with event image, title, venue, date, price
- **Interactive buttons**: 👍 Like, 👎 Dislike, 🎟️ Going, ⭐ Save
- **Swipe gestures** for mobile (swipe right = like, left = dislike)
- **Real-time updates** when user interacts with events

### 2. 🎛️ Smart Filtering System
- **Genre selection**: techno, house, indie, jazz, rock, electronic, etc.
- **Price range slider**: 0-1000 DKK with visual feedback
- **Date filters**: Tonight, Tomorrow, This Weekend, Next Week, This Month
- **Neighborhood picker**: Vesterbro, Nørrebro, Indre By, Christiania, Østerbro
- **Distance radius**: 1-25km from user location
- **Quick filter chips** for common combinations

### 3. 🆕 User Onboarding (Cold Start)
**3-step wizard for new users:**
- **Step 1**: Music preference selection (genre grid with icons)
- **Step 2**: Price range & location setup with map
- **Step 3**: Rate 5-10 sample events to bootstrap recommendations

### 4. 🔍 Search & Discovery
- **Smart search bar** with autocomplete suggestions
- **Search by**: venue name, artist, event title, neighborhood
- **Recent searches** with quick access
- **Trending searches** in Copenhagen
- **Voice search** support (optional)

### 5. 📍 Event Details & Map
- **Full event page** with description, lineup, venue details
- **Interactive map** showing venue location and nearby transport
- **Similar events** recommendations
- **Social sharing** buttons
- **Add to calendar** functionality
- **Venue information** and upcoming events

### 6. 👤 User Profile & History
- **Preferences management** (edit genres, price, location)
- **Interaction history**: Liked events, Going to, Saved events
- **Recommendation explanations**: "Why was this recommended?"
- **Stats**: Events attended, favorite venues, top genres

## 🎨 Design Requirements

### Visual Theme: Copenhagen Nightlife
- **Background**: Deep dark (#0a0a0a) with subtle gradients
- **Accent colors**: Neon cyan (#00f5ff), pink (#ff0080), green (#39ff14)
- **Cards**: Dark glass effect with subtle borders and blur
- **Typography**: Inter for body, Poppins for headings
- **Icons**: Lucide React with consistent styling

### UI Patterns
- **Glassmorphism**: Semi-transparent cards with backdrop blur
- **Neon accents**: Glowing borders and hover effects
- **Smooth animations**: 300ms transitions, easing curves
- **Copenhagen neighborhoods**: Color-coded (Vesterbro=pink, Nørrebro=green, etc.)
- **Mobile-first**: Touch-friendly buttons, swipe gestures

### Component Library
```typescript
// Key components to create:
- EventCard: Main event display with interactions
- FilterPanel: Collapsible filter interface  
- EventFeed: Infinite scroll container
- OnboardingWizard: 3-step preference collection
- SearchBar: Search with autocomplete
- MapView: Event location visualization
- UserProfile: Settings and history
- NavigationBar: Bottom tab navigation
```

## 🔌 API Integration

### Authentication
```typescript
// Anonymous users with localStorage UUID
const userId = localStorage.getItem('user_id') || generateUUID();
// Include in headers: 'X-User-ID': userId
```

### Key Endpoints
```typescript
// Get personalized recommendations
POST /recommend
{
  "user_preferences": {
    "preferred_genres": ["techno", "house"],
    "price_range": [100, 500],
    "location_lat": 55.6761,
    "location_lon": 12.5683
  },
  "num_recommendations": 20,
  "filters": {
    "date_filter": "this_weekend",
    "neighborhoods": ["Vesterbro", "Nørrebro"]
  }
}

// Record interactions
POST /interactions
{
  "event_id": "event_123",
  "interaction_type": "like", // like|dislike|going|save
  "source": "feed",
  "position": 3
}

// Search events
POST /search
{
  "query": "techno tonight vesterbro",
  "location_lat": 55.6761,
  "location_lon": 12.5683,
  "limit": 50
}
```

## 📱 User Experience Flows

### First-Time User Journey
1. **Landing page**: App intro with Copenhagen nightlife imagery
2. **Location permission**: Request with explanation of benefits
3. **Onboarding wizard**: 3-step preference collection
4. **First recommendations**: Curated events with explanations
5. **Tutorial tooltips**: Show interaction buttons and features

### Daily Usage Flow
1. **Open app**: Immediate personalized feed
2. **Quick filters**: "Tonight" or "This Weekend" buttons
3. **Browse & interact**: Swipe or tap to like/dislike events
4. **Event details**: Tap for full information and booking
5. **Social sharing**: Share interesting events

### Discovery & Exploration
1. **Search functionality**: Find specific events or venues
2. **Map exploration**: Browse events geographically  
3. **Filter experimentation**: Try different genre combinations
4. **Venue discovery**: Explore new locations and neighborhoods

## 🏗️ Technical Implementation

### State Management
```typescript
// Global state with Zustand
interface AppState {
  user: UserProfile;
  events: RecommendedEvent[];
  filters: FilterState;
  preferences: UserPreferences;
  location: UserLocation;
}
```

### Data Fetching
```typescript
// SWR for caching and real-time updates
const { data: events, mutate } = useSWR('/recommend', fetcher);
const { data: userProfile } = useSWR(`/users/${userId}`, fetcher);
```

### Performance Optimizations
- **Virtual scrolling** for long event lists
- **Image lazy loading** with placeholder gradients
- **API response caching** with SWR
- **Optimistic updates** for interactions
- **Service worker** for offline saved events

## 🌍 Copenhagen-Specific Features

### Local Context
- **Neighborhood character**: Vesterbro (trendy), Nørrebro (underground), Indre By (central)
- **Transport integration**: Show Metro/S-train stations near venues
- **Local venues**: Vega, Rust, Culture Box, Loppen, KB18 recognition
- **Cultural events**: Festivals, seasonal parties, underground scenes

### Language Support
- **Primary**: English interface
- **Secondary**: Danish text recognition for event descriptions
- **Local terms**: "aften" (evening), "fest" (party), "koncert" (concert)

## 📊 Analytics & Feedback

### User Behavior Tracking
- **Interaction rates**: Like/dislike ratios per event type
- **Session duration**: Time spent browsing events
- **Conversion tracking**: View → Going → Attended
- **A/B testing**: Different recommendation algorithms

### Recommendation Explanations
```typescript
// Show users why events were recommended
interface Explanation {
  overall_score: number;
  reasons: [
    "Strong match with your techno preferences",
    "Popular venue you've liked before", 
    "Close to your location",
    "Good price range for you"
  ];
  confidence: number;
}
```

## 🚀 Progressive Enhancement

### Core Features (MVP)
- Event feed with basic filtering
- Like/dislike interactions
- Simple search functionality
- User preferences management

### Enhanced Features
- Map view with clustering
- Social features (friends attending)
- Advanced recommendation explanations
- Offline functionality for saved events

### Future Enhancements
- AR venue discovery
- Real-time event updates
- Social group planning
- Integration with calendar apps

## 📋 Success Metrics

### User Engagement
- **Daily active users** returning to discover events
- **Interaction rate**: % of events users engage with
- **Session depth**: Average events viewed per session
- **Conversion rate**: Recommendations → event attendance

### Recommendation Quality
- **Relevance score**: User feedback on recommendations
- **Diversity satisfaction**: Variety in recommended events
- **Discovery rate**: Finding new venues/genres through app
- **Retention**: Users returning week over week

---

**Generate this as a modern, performant React app with the Copenhagen nightlife aesthetic and seamless API integration. Focus on mobile-first design and smooth animations that make event discovery fun and engaging!**