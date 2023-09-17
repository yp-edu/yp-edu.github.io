---
layout: home
permalink: /
---

## Welcome!

I am a French trainee researcher in AI, with a focus on AI Safety. I've had diverse experience in research,
education and industry. I'll outline this in a forthcoming story "A convoluted study path" giving insight in my career aspiration.
More on me in the [about](/about) section. 

You'll find here a collection of stories, projects and articles.
- Story: personnal thought, claim, discussion or experience 
- Project: concrete application (mostly around AI)
- Article: paper explaination or discussion (definitely about AI)

## Pinned [stories](/stories/)  [::rss::](/stories/feed.xml)

{% assign pinned_stories = "/stories/why-this-blog" | split: ", " %}
{% for story in site.stories %}
{% if pinned_stories contains story.url %}
### [{{ story.title }}](story.url)
{{ story.publishedOn }} | {{ story.tldr }}
{% endif %}
{% endfor %}

## Pinned [projects](/projects/) [::rss::](/projects/feed.xml)

Nothing yet..

## Pinned [articles](/articles/) [::rss::](/articles/feed.xml)

Nothing yet..