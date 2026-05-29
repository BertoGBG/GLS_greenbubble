// Open all external links in a new tab.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("a.reference.external").forEach(function (link) {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });
});
