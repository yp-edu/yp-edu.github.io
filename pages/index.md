---
layout: home
permalink: /
---

## Welcome!

I am a French trainee researcher in AI with a focus on AI Safety. I've had diverse experience in research, education and industry. 
I'll outline this in a forthcoming story, "A Convoluted Study Path", giving insight into my career aspiration. 
More on me in the [About](/about) section.

You'll find here a collection of stories, projects and articles.

- Story: personal thought, claim, discussion or experience 
- Project: concrete application (mostly around AI)
- Article: paper explanation or discussion (definitely about AI)

## Pinned [stories](/stories/)  [::rss::](/stories/feed.xml)

{% assign pinned_stories = "/stories/why-this-blog, /stories/my-approach-to-ai-safety, /stories/creating-a-startup" | split: ", " %}
{% assign collection = site.stories | sort: "publishedOn" | reverse %}
{% for item in collection %}
{% if pinned_stories contains item.url %}

### [{{ item.title }}]({{ item.url }})
<div class="thumbnail">
  {% if item.image %}
    <a href="{{ item.url }}"><img src="{{ item.image }}" alt="{{ item.title }}" class="thumbnail" /></a>
  {% endif %}
  {% if item.tldr %}
  <p class="tldr">
    <small class="date">{{ item.publishedOn }} </small><small>| {{ item.readingTime }} min read</small><small> | {{ item.tldr }}</small>
  </p>
  {% endif %}
</div>
{% endif %}
{% endfor %}

## Pinned [projects](/projects/) [::rss::](/projects/feed.xml)

Nothing yet..

## Pinned [articles](/articles/) [::rss::](/articles/feed.xml)

Nothing yet..
