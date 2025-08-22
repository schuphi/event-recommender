# 🚀 Lovable Integration Guide for Copenhagen Event Recommender

## Overview
This guide will help you integrate Lovable to generate the React frontend for our Copenhagen Event Recommender, then customize it with our advanced features.

## Prerequisites
✅ FastAPI backend running on `http://localhost:8000`  
✅ Database initialized with sample data  
✅ OpenAPI schema available at `http://localhost:8000/docs`  

## Step-by-Step Integration

### Step 1: Prepare Lovable Input

#### 1.1 Export OpenAPI Schema
```bash
# Start the FastAPI backend
cd backend
python -m uvicorn app.main:app --reload

# Export OpenAPI schema
curl http://localhost:8000/openapi.json > openapi.json
```

#### 1.2 Create Project Brief for Lovable
Create this prompt for Lovable:

```markdown
# Copenhagen Event Recommender - Frontend

## Project Description
Build a modern React frontend for a Copenhagen nightlife event recommendation app. The app uses ML to suggest events based on user preferences and behavior.

## API Integration
- Import the provided OpenAPI schema (openapi.json)
- Base URL: http://localhost:8000
- Authentication: Optional JWT + X-User-ID header for anonymous users
- Key endpoints: /recommend, /events, /interactions, /search

## Core Features to Implement

### 1. Event Discovery Feed
- Infinite scroll list of recommended events
- Card-based layout with event image, title, venue, date, price
- Real-time interaction buttons: 👍 Like, 👎 Dislike, 🎟️ Going, ⭐ Save
- Swipe gestures for mobile interactions

### 2. Advanced Filtering
- Genre selection (techno, house, indie, jazz, etc.)
- Price range slider (0-1000 DKK)
- Date filters (tonight, this weekend, next week)
- Neighborhood selection (Vesterbro, Nørrebro, Indre By, etc.)
- Distance radius from user location

### 3. User Onboarding (Cold Start)
- 3-step preference collection for new users
- Genre preferences selection
- Price range and location setup
- Sample event rating to kickstart recommendations

### 4. Search & Discovery
- Search bar with autocomplete
- Filter by venue, artist, or event type
- Recent searches and suggestions
- Voice search support

### 5. Event Details
- Full event page with description, lineup, venue info
- Interactive map showing venue location
- Similar events recommendations
- Social sharing buttons
- Add to calendar functionality

### 6. User Profile & Settings
- View and edit preferences
- Interaction history (liked, going, saved events)
- Recommendation explanation ("Why this event?")
- Privacy settings and data management

## Design Requirements

### Visual Style
- **Theme**: Dark Copenhagen nightlife aesthetic
- **Colors**: Dark background (#0a0a0a) with neon accents (cyan, pink, green)
- **Typography**: Modern sans-serif (Inter) with display font (Poppins)
- **Layout**: Card-based, mobile-first responsive design

### UI Components
- Glassmorphism effects for cards and modals
- Smooth animations and transitions
- Neon glow effects for interactive elements
- Copenhagen-inspired color palette
- High contrast for accessibility

### Mobile Experience
- Touch-friendly interaction buttons
- Swipe gestures for event cards
- Bottom sheet modals for filters
- Native-feeling scrolling and animations
- Offline capability for saved events

## Technical Stack
- **Framework**: Next.js 14 with TypeScript
- **Styling**: TailwindCSS with custom Copenhagen theme
- **State Management**: Zustand for global state
- **API Client**: SWR for data fetching and caching
- **Maps**: Leaflet with OpenStreetMap tiles
- **Icons**: Lucide React
- **Animations**: Framer Motion

## Key User Flows

### 1. First-Time User
1. Landing page with app introduction
2. Location permission request
3. 3-step preference onboarding
4. First recommendations with explanations
5. Tutorial overlay showing interaction buttons

### 2. Returning User
1. Personalized event feed based on history
2. Quick filters for tonight/weekend events
3. Notification of new events matching preferences
4. Social features (friends going to same events)

### 3. Event Discovery
1. Browse recommended events in feed
2. Use filters to refine results
3. View event details and venue information
4. Interact with events (like, going, save)
5. Share interesting events with friends

## API Integration Details

### Authentication
```typescript
// Anonymous user with localStorage UUID
const userId = localStorage.getItem('user_id') || generateUUID();
headers['X-User-ID'] = userId;
```

### Core API Calls
```typescript
// Get personalized recommendations
POST /recommend
{
  "user_preferences": { "preferred_genres": ["techno", "house"] },
  "location_lat": 55.6761,
  "location_lon": 12.5683,
  "num_recommendations": 20
}

// Record user interaction
POST /interactions
{
  "event_id": "event_123",
  "interaction_type": "like",
  "source": "feed",
  "position": 3
}

// Search events
POST /search
{
  "query": "techno tonight",
  "location_lat": 55.6761,
  "location_lon": 12.5683
}
```

## Copenhagen-Specific Features

### Local Context
- Neighborhood-aware recommendations (Vesterbro vibes vs Nørrebro underground)
- Danish language support for event descriptions
- Local venue recognition and popularity
- Public transport integration (Metro/S-train to venues)

### Cultural Elements
- Recognition of Copenhagen nightlife culture
- Underground/warehouse party detection
- Festival and seasonal event awareness
- Local artist and venue prioritization

## Performance Requirements
- Initial load < 2 seconds
- Smooth 60fps scrolling and animations
- Lazy loading for event images
- Offline support for saved events
- Progressive Web App capabilities

## Accessibility
- WCAG 2.1 AA compliance
- Screen reader support
- High contrast mode
- Keyboard navigation
- Touch target sizes (44px minimum)

## Post-Lovable Customization Plan
After Lovable generates the base app, we'll add:
1. Advanced ML explanation UI
2. Social features and user connections
3. Admin dashboard for event management
4. Real-time viral event detection
5. Analytics and A/B testing framework
```

### Step 2: Setup Lovable Project

#### 2.1 Create New Lovable Project
1. Go to [Lovable.dev](https://lovable.dev)
2. Create new project: "Copenhagen Event Recommender"
3. Choose React + TypeScript template

#### 2.2 Upload Prepared Files
Upload these files to Lovable:
- `openapi.json` (exported schema)
- Project brief (from above)
- `types/api.ts` (TypeScript definitions)
- `lib/api.ts` (API client)
- `tailwind.config.js` (Copenhagen theme)

### Step 3: Generate Base Frontend

#### 3.1 Initial Generation
Provide Lovable with:
```
Generate a Copenhagen nightlife event discovery app with:
1. Event feed with recommendation cards
2. Like/dislike/going interaction buttons  
3. Filter panel for genre/price/location
4. User onboarding flow
5. Search functionality
6. Event detail pages
7. Dark theme with neon accents

Use the provided OpenAPI schema for all API integration.
```

#### 3.2 Iterate on Design
Request improvements:
- "Make the event cards more visually appealing with glassmorphism"
- "Add smooth animations for interactions"
- "Improve the filter UI with better Copenhagen neighborhood selection"
- "Add map view integration"

### Step 4: Post-Lovable Customization

#### 4.1 Advanced Features to Add
```typescript
// Enhanced event card with ML explanations
const EventCardEnhanced = ({ event, explanation }) => {
  return (
    <div className="event-card glass-effect">
      {/* Lovable base card */}
      <ExplanationTooltip reasons={explanation.reasons} />
      <ViralIndicator isViral={event.source === 'viral'} />
      <SwipeGestures onSwipe={handleInteraction} />
    </div>
  );
};

// Real-time recommendation updates
const useRealtimeRecommendations = () => {
  // WebSocket connection for live updates
  // Social proof notifications
  // Viral event alerts
};
```

#### 4.2 Copenhagen-Specific Enhancements
```typescript
// Neighborhood color coding
const neighborhoodColors = {
  'Vesterbro': '#ff0080', // Pink
  'Nørrebro': '#39ff14',  // Green  
  'Indre By': '#00f5ff',  // Cyan
  'Christiania': '#8a2be2' // Purple
};

// Local venue recognition
const VenueCard = ({ venue }) => (
  <div className={`venue-card ${getVenueStyle(venue.name)}`}>
    <VenueReputation score={venue.popularity} />
    <TransportLinks venue={venue} />
  </div>
);
```

### Step 5: Integration & Testing

#### 5.1 Connect to Backend
```bash
# Update environment variables
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Test API connection
npm run dev
# Verify recommendations load correctly
```

#### 5.2 Add Advanced Features
1. **ML Explanations**: Why events were recommended
2. **Viral Detection**: Highlight trending events from social scraping
3. **Social Features**: See what friends are attending
4. **Analytics**: Track user behavior and recommendation performance

#### 5.3 Performance Optimization
- Image optimization and lazy loading
- API response caching with SWR
- Virtual scrolling for long event lists
- Service worker for offline capability

### Step 6: Deployment

#### 6.1 Vercel Deployment (Lovable can help)
```bash
# Lovable can generate deployment config
vercel --prod
```

#### 6.2 Environment Setup
```bash
# Production environment
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_ANALYTICS_ID=your_analytics_id
```

## Expected Timeline

**Lovable Generation**: 2-4 hours
- Base frontend with all core features
- API integration complete
- Responsive design implemented

**Post-Lovable Customization**: 1-2 days  
- Copenhagen-specific styling
- Advanced ML features
- Performance optimization
- Analytics integration

## Benefits of This Approach

✅ **Rapid Development**: Working app in hours vs weeks  
✅ **Professional Quality**: Modern React best practices  
✅ **Type Safety**: Full TypeScript integration  
✅ **API Integration**: Seamless backend connection  
✅ **Customization Ready**: Clean codebase for our enhancements  

## Next Steps

1. **Prepare the OpenAPI export** (5 minutes)
2. **Create Lovable project with brief** (15 minutes)  
3. **Generate and iterate on frontend** (2-4 hours)
4. **Add Copenhagen-specific features** (1-2 days)
5. **Deploy and test** (30 minutes)

This hybrid approach gives us a production-ready frontend foundation while preserving the ability to add our unique Copenhagen nightlife features and advanced ML capabilities!