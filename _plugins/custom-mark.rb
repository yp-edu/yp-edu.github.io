
require "jekyll-commonmark"
require 'kramdown'

FA_CALLOUTS = {
  "tip" => "fa-solid fa-fire-flame-curved",
  "hint" => "fa-solid fa-fire-flame-curved",
  "important" => "fa-solid fa-fire-flame-curved",
  "abstract" => "fa-solid fa-clipboard-list",
  "summary" => "fa-solid fa-clipboard-list",
  "tldr" => "fa-solid fa-clipboard-list",
  "info" => "fa-solid fa-circle-info",
  "bug" => "fa-solid fa-bug",
  "note" => "fa-solid fa-pen",
  "rss" => "fa-solid fa-rss",
  "linkedin" => "fa-brands fa-linkedin",
  "github" => "fa-brands fa-github",
  "warning" => "fa-solid fa-triangle-exclamation",
  "caution" => "fa-solid fa-triangle-exclamation",
  "success" => "fa-solid fa-check",
  "danger" => "fa-solid fa-bolt",
  "error" => "fa-solid fa-bolt",
  "example" => "fa-solid fa-list",
  "quote" => "fa-solid fa-quote-right",
  "question" => "fa-solid fa-circle-question",
  "answer" => "fa-solid fa-circle-info",
  "todo" => "fa-solid fa-circle-check",
  "thanks" => "fa-solid fa-heart",
  "fail" => "fa-solid fa-xmark",
}
DEBUG = false

class Jekyll::Converters::Markdown
  class CustomMarkProcessor < CommonMark
    def convert(content)
      content = content.gsub(/\%\\label\{/, '\label{')
      content = content.gsub(/```/, '~~~')
      options = {
        syntax_highlighter: 'rouge',
        syntax_highlighter_opts: {
          css_class: 'highlight',
          block: {
            line_numbers: true
          }
        }
      }
      html = Kramdown::Document.new(content, options).to_html
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "content class: #{content.class}"
        Jekyll.logger.info "CustomMarkProcessor:", "html class: #{html.class}"
      end
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "html: #{html}"
      end
      # regexp = /<blockquote>\s*.*<p>\[\!([a-z]{1,})\]\s?(.*)<\/p>\s*.*<p>(.*)<\/p>\s*.*<\/blockquote>/
      # regexp = /<blockquote>\s<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s<p>((?:(?!blockquote).)*)<\/p>\s<\/blockquote>/
      # regexp = /<blockquote>\s*<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s*<p>((?:(?!blockquote).)*)<\/p>\s*<\/blockquote>/m
      # matches = html.match(regexp)
      
      parsed_html = html.gsub(
        /<blockquote>\s*<p>\[\!(answer)\]((?:(?!blockquote).)*)<\/p>\s*<p>((?:(?!blockquote).)*)<\/p>\s*<\/blockquote>/m, 
        '<blockquote class="callout \1"> <details> <summary><div class="callout-title"> '\
        '<i class="fa-\1" href="#"></i> '\
        '<em>\2</em></div></summary> <p>\3</p> </details> </blockquote>'
        )
      parsed_html = parsed_html.gsub(
        /<blockquote>\s*<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s*<p>((?:(?!blockquote).)*)<\/p>\s*<\/blockquote>/m, 
        '<blockquote class="callout \1"> <div class="callout-title"> '\
        '<i class="fa-\1" href="##"></i> '\
        '<em>\2</em></div> <p>\3</p> </blockquote>'
        )
      parsed_html = parsed_html.gsub(
        /<blockquote>\s*<p>\[\!((?:example)|(?:answer))\]((?:(?!blockquote).)*)<\/p>\s*<ul>((?:(?!blockquote).)*)<\/ul>\s*<\/blockquote>/m,
        '<blockquote class="callout \1"> <details> <summary><div class="callout-title"> '\
        '<i class="fa-\1" href="#"></i> '\
        '<em>\2</em></div></summary> <ul>\3</ul> </details> </blockquote>'
        )
      parsed_html = parsed_html.gsub(
          /<blockquote>\s*<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s*<p>((?:(?!blockquote).)*)<\/p>\s*<ul>((?:(?!blockquote).)*)<\/ul>\s*<\/blockquote>/m,
          '<blockquote class="callout \1"> <div class="callout-title"> '\
          '<i class="fa-\1" href="#"></i> '\
          '<em>\2</em></div> <p>\3</p> <ul>\4</ul> </blockquote>'
          )
      parsed_html = parsed_html.gsub(
        /<blockquote>\s*<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s*<ul>((?:(?!blockquote).)*)<\/ul>\s*<\/blockquote>/m,
        '<blockquote class="callout \1"> <div class="callout-title"> '\
        '<i class="fa-\1" href="#"></i> '\
        '<em>\2</em></div> <ul>\3</ul> </blockquote>'
        )
      parsed_html = parsed_html.gsub(
        /<blockquote>\s*<p>\[\!([a-z]{1,})\]((?:(?!blockquote).)*)<\/p>\s*<ol>((?:(?!blockquote).)*)<\/ol>\s*<\/blockquote>/m,
        '<blockquote class="callout \1"> <div class="callout-title"> '\
        '<i class="fa-\1" href="#"></i> '\
        '<em>\2</em></div> <ol>\3</ol> </blockquote>'
        )

      parsed_html = parsed_html.gsub(/::((?:(?!\:\:).)*)::/, '<i class="fa-\1"></i>')
      FA_CALLOUTS.each do |key, value|
        parsed_html.gsub!(/<i class="fa-#{key}"/, "<i class=\"#{value}\"")
      end
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "matches: #{matches}"
        Jekyll.logger.info "CustomMarkProcessor:", "parsed_html: #{parsed_html}"
      end
      #parsed_html = parsed_html.gsub(/<h([1-6])\sid="([a-z]|\-)*">/, '<h\1>')
      parsed_html = parsed_html.gsub(/<h2(\sid="([a-z]|\-)*")?>(\s*.*)<\/h2>/, '<h2\1># \3</h2>')

      parsed_html = parsed_html.gsub(/<img(.*)\ssrc="\.(.*)\/((?:(?!\/).)*)"(.*)/, '<img\1 src="/assets/images/\3"\4')
      parsed_html = parsed_html.gsub(/<img(.*)\ssrc="((?:(?!\/).)*)"(.*)/, '<img\1 src="/assets/images/\2"\3')
      parsed_html
    end
  end
end